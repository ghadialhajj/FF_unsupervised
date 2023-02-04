import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from prepare_data import prepare_data


def goodness_score(pos_acts, neg_acts, threshold=2):
    pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    return torch.add(pos_goodness, neg_goodness)


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, n_epochs: int, bias: bool, device):
        super().__init__(in_features, out_features, bias=bias)
        self.n_epochs = n_epochs
        self.opt = torch.optim.Adam(self.parameters())
        self.goodness = goodness_score
        self.to(device)
        self.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)

    def ff_train(self, pos_acts, neg_acts):
        self.opt.zero_grad()
        goodness = self.goodness(pos_acts, neg_acts)
        goodness.backward()
        self.opt.step()

    def forward(self, input):
        input = super().forward(input)
        input = self.ln_layer(input.detach())
        return input


class Unsupervised_FF(nn.Module):
    def __init__(self, n_layers: int = 4, n_neurons=2000, input_size: int = 28 * 28, n_epochs: int = 100,
                 bias: bool = True, n_classes: int = 10, n_hid_to_log: int = 3, device=torch.device("cuda:0")):
        super().__init__()
        self.n_hid_to_log = n_hid_to_log
        self.n_epochs = n_epochs
        self.device = device

        ff_layers = [
            FF_Layer(in_features=input_size if idx == 0 else n_neurons,
                     out_features=n_neurons,
                     n_epochs=n_epochs,
                     bias=bias,
                     device=device) for idx in range(n_layers)]

        self.ff_layers = ff_layers
        self.last_layer = nn.Linear(in_features=n_neurons * n_hid_to_log, out_features=n_classes, bias=bias)
        self.to(device)
        self.opt = torch.optim.Adam(self.last_layer.parameters())
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def train_ff_layers(self, pos_dataloader, neg_dataloader):
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training FF Layers", position=0)
        for epoch in outer_tqdm:
            inner_tqdm = tqdm(zip(pos_dataloader, neg_dataloader), desc=f"Training FF Layers | Epoch {epoch}",
                              leave=False, position=1)
            for pos_data, neg_imgs in inner_tqdm:
                pos_imgs, _ = pos_data
                pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(device)
                neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(device)

                for idx, layer in enumerate(self.ff_layers):
                    pos_acts = layer(pos_acts)
                    neg_acts = layer(neg_acts)
                    layer.ff_train(pos_acts, neg_acts)

    def train_last_layer(self, dataloader: DataLoader):
        num_examples = len(dataloader)
        outer_tqdm = tqdm(range(self.n_epochs), desc="Training Last Layer", position=0)
        loss_list = []
        for epoch in outer_tqdm:
            epoch_loss = 0
            inner_tqdm = tqdm(dataloader, desc=f"Training Last Layer | Epoch {epoch}", leave=False, position=1)
            for images, labels in inner_tqdm:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.opt.zero_grad()
                preds = self(images)
                loss = self.loss(preds, labels)
                epoch_loss += loss
                loss.backward()
                self.opt.step()
            loss_list.append(epoch_loss / num_examples)
            # Update progress bar with current loss
        return [l.detach().cpu().numpy() for l in loss_list]

    def forward(self, image: torch.Tensor):
        image = image.to(self.device)
        image = torch.reshape(image, (image.shape[0], 1, -1))
        concat_output = []
        for idx, layer in enumerate(self.ff_layers):
            image = layer(image)
            if idx > len(self.ff_layers) - self.n_hid_to_log - 1:
                concat_output.append(image)
        concat_output = torch.concat(concat_output, 2)
        logits = self.last_layer(concat_output)
        return logits.squeeze()


def train(model: Unsupervised_FF, pos_dataloader: DataLoader, neg_dataloader: DataLoader):
    model.train_ff_layers(pos_dataloader, neg_dataloader)
    return model.train_last_layer(pos_dataloader)


if __name__ == '__main__':
    prepare_data()
    device = torch.device("cuda:0")
    unsupervised_ff = Unsupervised_FF(device=device, n_epochs=100)
    # Load the MNIST dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    pos_dataset = torchvision.datasets.MNIST(root='train_data/', download=True, transform=transform)
    # pos_dataset = Subset(pos_dataset, list(range(1000)))
    # Create the data loader
    pos_dataloader = DataLoader(pos_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Load the transformed images
    neg_dataset = torch.load('transformed_dataset.pt')
    # neg_dataset = Subset(neg_dataset, list(range(1000)))
    # Create the data loader
    neg_dataloader = DataLoader(neg_dataset, batch_size=64, shuffle=True, num_workers=4)

    loss_list = train(unsupervised_ff, pos_dataloader, neg_dataloader)
    fig = plt.figure()
    plt.plot(loss_list)
    plt.savefig("Loss.png")
