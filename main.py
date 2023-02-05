import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import prepare_data
from sklearn.metrics import accuracy_score


def goodness_score(pos_acts, neg_acts, threshold=2):
    """
    Compute the goodness score for a given set of positive and negative activations.

    Parameters:

    pos_acts (torch.Tensor): Numpy array of positive activations.
    neg_acts (torch.Tensor): Numpy array of negative activations.
    threshold (int, optional): Threshold value used to compute the score. Default is 2.

    Returns:

    goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
    score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
    quantity but without the threshold subtraction
    """

    pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
    neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
    return torch.add(pos_goodness, neg_goodness)


def get_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    return dict(accuracy_score=acc)


class FF_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, n_epochs: int, bias: bool, device):
        super().__init__(in_features, out_features, bias=bias)
        self.n_epochs = n_epochs
        self.opt = torch.optim.Adam(self.parameters())
        self.goodness = goodness_score
        self.to(device)
        self.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)

    def ff_train(self, pos_acts, neg_acts):
        """
        Train the layer using positive and negative activations.

        Parameters:

        pos_acts (numpy.ndarray): Numpy array of positive activations.
        neg_acts (numpy.ndarray): Numpy array of negative activations.
        """

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
                pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(self.device)
                neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(self.device)

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

    def evaluate(self, dataloader: DataLoader, dataset_type: str = "train"):
        self.eval()
        inner_tqdm = tqdm(dataloader, desc=f"Evaluating model", leave=False, position=1)
        all_labels = []
        all_preds = []
        for images, labels in inner_tqdm:
            images = images.to(self.device)
            labels = labels.to(self.device)
            preds = self(images)
            preds = torch.argmax(preds, 1)
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
        all_labels = torch.concat(all_labels, 0).numpy()
        all_preds = torch.concat(all_preds, 0).numpy()
        metrics_dict = get_metrics(all_preds, all_labels)
        print(f"{dataset_type} dataset scores: ", "\n".join([f"{key}: {value}" for key, value in metrics_dict.items()]))


def train(model: Unsupervised_FF, pos_dataloader: DataLoader, neg_dataloader: DataLoader):
    model.train()
    model.train_ff_layers(pos_dataloader, neg_dataloader)
    return model.train_last_layer(pos_dataloader)


def plot_loss(loss):
    # plot the loss over epochs
    fig = plt.figure()
    plt.plot(list(range(len(loss))), loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.savefig("Loss Plot.png")
    plt.show()


if __name__ == '__main__':
    prepare_data()

    # Load the MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    pos_dataset = torchvision.datasets.MNIST(root='./train_data/', download=False, transform=transform)
    # pos_dataset = Subset(pos_dataset, list(range(1000)))
    # Create the data loader
    pos_dataloader = DataLoader(pos_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Load the transformed images
    neg_dataset = torch.load('transformed_dataset.pt')
    # Create the data loader
    neg_dataloader = DataLoader(neg_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Load the test images
    test_dataset = torchvision.datasets.MNIST(root='./test_data/', train=False, download=False, transform=transform)
    # Create the data loader
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    device = torch.device("cuda:0")
    unsupervised_ff = Unsupervised_FF(device=device, n_epochs=2)

    loss = train(unsupervised_ff, pos_dataloader, neg_dataloader)

    plot_loss(loss)

    unsupervised_ff.evaluate(pos_dataloader, dataset_type="Train")
    unsupervised_ff.evaluate(test_dataloader, dataset_type="Test")
