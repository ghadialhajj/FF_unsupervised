import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils import create_negative_image
from tqdm import tqdm
import os


def prepare_data():
    # Define the transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the train MNIST dataset
    train_mnist_dataset = torchvision.datasets.MNIST(root='train_data/', train=True, transform=transform,
                                                     download=True)
    n_train_samples = len(train_mnist_dataset)
    # Load the test MNIST dataset
    test_mnist_dataset = torchvision.datasets.MNIST(root='test_data/', train=False, transform=transform,
                                                    download=True)

    if not os.path.exists("transformed_dataset.pt"):
        random_pairs = np.random.randint(n_train_samples, size=[n_train_samples, 2])
        random_pairs = [(row[0], row[1]) for row in random_pairs]

        # Transform the data
        transformed_dataset = [
            create_negative_image(train_mnist_dataset[pair[0]][0].squeeze(), train_mnist_dataset[pair[1]][0].squeeze()) for
            pair in tqdm(random_pairs)]

        # Save the transformed images to a folder
        torch.save(transformed_dataset, 'transformed_dataset.pt')

    # Load the transformed images
    transformed_dataset = torch.load('transformed_dataset.pt')

    # Create a dataset from the transformed images
    dataset = torch.utils.data.TensorDataset(torch.stack(transformed_dataset))

    # Create a dataloader from the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader


if __name__ == '__main__':
    dl = prepare_data()
    img = next(iter(dl))
    print(next(iter(dl)))
