
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from tqdm.auto import tqdm
import shutil
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch import tensor, Tensor

######################################################################################

os.system('cls || clear')

######################################################################################

epochs = 200
batch_size = 64
n_neurons = 2000
n_classes = 10
n_layers = 4
input_size = 28 * 28
n_hid_to_log = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################################

def clean_repo():
	"""
	Deletes the 'MNIST' folder and the 'transformed_dataset.pt' file if they exist in the working directory.
	"""
	folder_path = "MNIST"
	if os.path.exists(folder_path):
		shutil.rmtree(folder_path)

	file_path = "transformed_dataset.pt"
	if os.path.exists(file_path):
		os.remove(file_path)

######################################################################################

def prepare_data():
	"""
	Prepares the MNIST dataset by loading it, randomly pairing the images, creating negative images from the pairs,
	and saving them to 'transformed_dataset.pt' file.
	"""
	# Define the transform function
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	# Load the train MNIST dataset
	train_mnist_dataset = torchvision.datasets.MNIST(root="./", train=True, transform=transform, download=True)
	n_train_samples = len(train_mnist_dataset)

	# Load the test MNIST dataset
	test_mnist_dataset = torchvision.datasets.MNIST(root="./", train=False, transform=transform, download=True)

	if not os.path.exists("transformed_dataset.pt"):
		random_pairs = np.random.randint(n_train_samples, size=[n_train_samples, 2])
		random_pairs = [(row[0], row[1]) for row in random_pairs]

		# Transform the data
		transformed_dataset = [create_negative_image(train_mnist_dataset[pair[0]][0].squeeze(), train_mnist_dataset[pair[1]][0].squeeze()) for pair in tqdm(random_pairs, desc='Preparing Dataset')]

		# Save the transformed images to a file
		torch.save(transformed_dataset, 'transformed_dataset.pt')

######################################################################################

def create_mask(shape, iterations: int = 10):
	"""
	Create a binary mask as described in (Hinton, 2022): start with a random binary image and then repeatedly blur
	the image with a filter, horizontally and vertically.

	Parameters
	----------
	shape : tuple
		The shape of the output mask (height, width).
	iterations : int
		The number of times to blur the image.

	Returns
	-------
	numpy.ndarray
		A binary mask with the specified shape, containing fairly large regions of ones and zeros.
	"""
	
	# Define the horizontal and vertical blur filters
	blur_filter_1 = np.array(((0, 0, 0), (1/4, 1/2, 1/4), (0, 0, 0)))
	blur_filter_2 = blur_filter_1.T

	# Create a random binary image
	image = np.random.randint(0, 2, size=shape)

	# Blur the image with the specified filter
	for i in range(iterations):
		image = np.abs(convolve2d(image, blur_filter_1, mode='same') / blur_filter_1.sum())
		image = np.abs(convolve2d(image, blur_filter_2, mode='same') / blur_filter_2.sum())

	# Binarize the blurred image, i.e. threshold it at 0.5
	mask = np.round(image).astype(np.uint8)

	return tensor(mask)

######################################################################################

def create_negative_image(image_1: Tensor, image_2: Tensor):
	"""
	Create a negative image by combining two images with a binary mask.

	Parameters:
	image_1 (Tensor): The first image to be combined.
	image_2 (Tensor): The second image to be combined.

	Returns:
	Tensor: The negative image created by combining the two input images.

	Raises:
	AssertionError: If the shapes of `image_1` and `image_2` are not the same.

	Examples:
	>>> image_1 = np.random.randint(0, 2, size=(5, 5))
	>>> image_2 = np.random.randint(0, 2, size=(5, 5))
	>>> create_negative_image(image_1, image_2)
	array([[0 0 0 0 1]
		[1 1 0 1 1]
		[0 0 0 1 1]
		[0 1 1 1 0]
		[1 1 0 0 1]])
	"""

	# Check if the shapes of the input images are the same
	assert image_1.shape == image_2.shape, "Incompatible images and mask shapes."

	# Create a binary mask
	mask = create_mask((image_1.shape[0], image_1.shape[1]))

	# Apply the binary mask to the two input images
	image_1 = torch.mul(image_1, mask)
	image_2 = torch.mul(image_2, 1 - mask)

	# Combine the masked images to form the negative image
	return torch.add(image_1, image_2)

######################################################################################

def goodness_score(pos_acts: Tensor, neg_acts: Tensor, threshold: float = 2.0) -> Tensor:
	"""
	Compute the goodness score for a given set of positive and negative activations.

	Parameters:
	pos_acts (torch.Tensor): Numpy array of positive activations.
	neg_acts (torch.Tensor): Numpy array of negative activations.
	threshold (float, optional): Threshold value used to compute the score. Default is 2.

	Returns:
	goodness (torch.Tensor): Goodness score computed as the sum of positive and negative goodness values. Note that this
	score is actually the quantity that is optimized and not the goodness itself. The goodness itself is the same
	quantity but without the threshold subtraction
	"""
	pos_goodness = -torch.sum(torch.pow(pos_acts, 2)) + threshold
	neg_goodness = torch.sum(torch.pow(neg_acts, 2)) - threshold
	return torch.add(pos_goodness, neg_goodness)

######################################################################################

def get_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
	"""
	Compute the accuracy score for a given set of predicted and actual labels.

	Parameters:
	preds (numpy.ndarray): Numpy array of predicted labels.
	labels (numpy.ndarray): Numpy array of actual labels.

	Returns:
	dict[str, float]: Dictionary containing the accuracy score.
	"""
	acc = accuracy_score(labels, preds)
	return dict(accuracy_score=acc)

######################################################################################

def ff_layer_init(in_features: int, out_features: int, n_epochs: int, bias: bool, device: torch.device) -> nn.Module:
	"""
	Initializes a fully connected layer with specified parameters.

	Parameters:
	in_features (int): Number of input features.
	out_features (int): Number of output features.
	n_epochs (int): Number of epochs for training.
	bias (bool): Whether to use bias or not.
	device (torch.device): Device to train the layer on.

	Returns:
	nn.Module: Initialized fully connected layer.
	"""
	layer = nn.Linear(in_features, out_features, bias=bias)
	layer.n_epochs = n_epochs
	layer.opt = torch.optim.Adam(layer.parameters())
	layer.goodness = goodness_score
	layer.to(device)
	layer.ln_layer = nn.LayerNorm(normalized_shape=[1, out_features]).to(device)
	return layer

######################################################################################

def ff_train(layer: nn.Module, pos_acts: Tensor, neg_acts: Tensor, goodness) -> None:
	"""
	Trains the specified fully connected layer.

	Parameters:
	layer (nn.Module): Fully connected layer to train.
	pos_acts (torch.Tensor): Numpy array of positive activations.
	neg_acts (torch.Tensor): Numpy array of negative activations.
	goodness: Function used to calculate the goodness score.

	Returns:
	None
	"""
	layer.opt.zero_grad()
	goodness = goodness(pos_acts, neg_acts)
	goodness.backward()
	layer.opt.step()

######################################################################################

def ff_forward(layer: nn.Module, input: Tensor) -> Tensor:
	"""
	Calculates forward pass of specified fully connected layer.

	Parameters:
	layer (nn.Module): Fully connected layer for which to calculate the forward pass.
	input (torch.Tensor): Input tensor for forward pass.

	Returns:
	Tensor: Output tensor of forward pass.
	"""
	input = layer(input)
	input = layer.ln_layer(input.detach())
	return input

######################################################################################

def supervised_ff_init(n_layers, bias, n_classes, n_hid_to_log, device, n_neurons, input_size, n_epochs):
	"""
	Initialize an supervised feed-forward neural network.

	Parameters:

	n_layers (int): The number of hidden layers in the neural network.
	bias (bool): If True, use bias terms in the neural network.
	n_classes (int): The number of classes to predict.
	n_hid_to_log (int): The number of hidden layers to log in the output layer.
	device (torch.device): The device to use for computations.
	n_neurons (int): The number of neurons in each hidden layer.
	input_size (int): The number of input features.
	n_epochs (int): The number of epochs to train for.

	Returns:

	nn.Module: The initialized neural network.
	"""

	model = nn.Module()
	model.n_hid_to_log = n_hid_to_log
	model.n_epochs = n_epochs
	model.device = device

	ff_layers = [ff_layer_init(in_features=input_size if idx == 0 else n_neurons,out_features=n_neurons, n_epochs=n_epochs, bias=bias, device=device) for idx in range(n_layers)]

	model.ff_layers = ff_layers
	model.last_layer = nn.Linear(in_features=n_neurons * n_hid_to_log, out_features=n_classes, bias=bias)

	model.to(device)
	model.opt = torch.optim.Adam(model.last_layer.parameters())
	model.loss = torch.nn.CrossEntropyLoss(reduction="mean")
	return model

######################################################################################

def train(model: nn.Module, pos_dataloader: DataLoader, neg_dataloader: DataLoader, goodness_score: float) -> list[float]:
	"""
	Train the supervised feedforward neural network model.

	Parameters:
	model (nn.Module): The supervised feedforward neural network model.
	pos_dataloader (DataLoader): A PyTorch DataLoader containing the positive images to be used for training.
	neg_dataloader (DataLoader): A PyTorch DataLoader containing the negative images to be used for training.
	goodness_score (float): The threshold to use when computing the goodness score.

	Returns:
	list[float]: A list containing the loss values for each epoch of training.
	"""
	train_ff_layers(model, pos_dataloader, neg_dataloader, goodness_score)
	return train_last_layer(model, pos_dataloader)

######################################################################################

def train_ff_layers(model: nn.Module, pos_dataloader: DataLoader, neg_dataloader: DataLoader, goodness_score: float):
	"""
	Train the feedforward layers of the supervised feedforward neural network model.

	Parameters:
	model (nn.Module): The supervised feedforward neural network model.
	pos_dataloader (DataLoader): A PyTorch DataLoader containing the positive images to be used for training.
	neg_dataloader (DataLoader): A PyTorch DataLoader containing the negative images to be used for training.
	goodness_score (float): The threshold to use when computing the goodness score.
	"""
	for epoch in tqdm(range(model.n_epochs), desc='Training FF Layers'):
		for pos_data, neg_imgs in zip(pos_dataloader, neg_dataloader):
			pos_imgs, _ = pos_data
			pos_acts = torch.reshape(pos_imgs, (pos_imgs.shape[0], 1, -1)).to(model.device)
			neg_acts = torch.reshape(neg_imgs, (neg_imgs.shape[0], 1, -1)).to(model.device)

			for idx, layer in enumerate(model.ff_layers):
				pos_acts = ff_forward(layer, pos_acts)
				neg_acts = ff_forward(layer, neg_acts)
				ff_train(layer, pos_acts, neg_acts, goodness_score)

######################################################################################

def train_last_layer(model: nn.Module, dataloader: DataLoader) -> list[float]:
	"""
	Train the last layer of the supervised feedforward neural network model.

	Parameters:
	model (nn.Module): The supervised feedforward neural network model.
	dataloader (DataLoader): A PyTorch DataLoader containing the data to be used for training.

	Returns:
	list[float]: A list containing the loss values for each epoch of training.
	"""
	loss_list = []
	for epoch in tqdm(range(model.n_epochs), desc='Training Last Layer'):
		epoch_loss = 0
		for images, labels in dataloader:
			images = images.to(model.device)
			labels = labels.to(model.device)
			model.opt.zero_grad()
			preds = supervised_ff_forward(model, images)
			loss = model.loss(preds, labels)
			epoch_loss += loss
			loss.backward()
			model.opt.step()
		loss_list.append(epoch_loss / len(dataloader))
	return [l.detach().cpu().numpy() for l in loss_list]

######################################################################################

def evaluate(model: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
	"""
	Evaluate the model on the test set.

	Parameters:
	model (nn.Module): The supervised feedforward neural network model to be evaluated.
	dataloader (DataLoader): The data loader for the test set.

	Returns:
	Tuple[float, float]: A tuple containing the accuracy of the model on the test set as a float and the number of
	correct predictions as an integer.
	"""

	nn.Module.eval(model)
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(model.device)
			labels = labels.to(model.device)
			outputs = supervised_ff_forward(model, images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return correct / total, correct

######################################################################################

def supervised_ff_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
	"""
	Forward pass through the supervised feedforward neural network.

	Parameters:
	model (nn.Module): The supervised feedforward neural network model.
	image (torch.Tensor): The input image tensor.

	Returns:
	torch.Tensor: The output tensor of the neural network.

	"""
	# Move the image tensor to the device
	image = image.to(model.device)

	# Reshape the image tensor to the correct dimensions
	image = torch.reshape(image, (image.shape[0], 1, -1))

	concat_output = []
	for idx, layer in enumerate(model.ff_layers):
		# Pass the input through the feedforward layer
		image = ff_forward(layer, image)

		# Store the output of the last n_hid_to_log layers
		if idx > len(model.ff_layers) - model.n_hid_to_log - 1:
			concat_output.append(image)

	# Concatenate the outputs of the last n_hid_to_log layers
	concat_output = torch.cat(concat_output, 2)

	# Pass the concatenated output through the last layer
	logits = model.last_layer(concat_output)

	return logits.squeeze()

######################################################################################

def plot_loss(loss):
	"""
	Plots the loss over epochs.

	Args:
		loss (list): A list of losses for each epoch.

	Returns:
		None
	"""
	# Create a new figure
	fig = plt.figure()

	# Plot the loss values against the number of epochs
	plt.plot(list(np.int_(range(len(loss)))), loss)

	# Add labels for the x and y axes and a title for the plot
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("Loss Plot")

	# Save the plot as an image file
	plt.savefig("Loss Plot.png", bbox_inches='tight', dpi=200)

	# Close the figure to free up memory
	plt.close()

######################################################################################

if __name__ == '__main__':

	file_path = "Loss Plot.png"
	if os.path.exists(file_path):
		os.remove(file_path)

	clean_repo()

	print()
	
	prepare_data()

	print()

	# Load the MNIST dataset
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

	pos_dataset = torchvision.datasets.MNIST(root='./', download=False, transform=transform, train=True)

	# Create the data loader
	pos_dataloader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	# Load the transformed images
	neg_dataset = torch.load('transformed_dataset.pt')

	# Create the data loader
	neg_dataloader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	# Load the test images
	test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=False, transform=transform)

	# Create the data loader
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	supervised_ff = supervised_ff_init(n_layers=n_layers, bias=True, n_classes=n_classes, n_hid_to_log=n_hid_to_log, device=device, n_neurons=n_neurons, input_size=input_size, n_epochs=epochs)

	loss = train(supervised_ff, pos_dataloader, neg_dataloader, goodness_score)

	plot_loss(loss)

	print()

	accuracy_train, correct_train = evaluate(supervised_ff, pos_dataloader)
	print(f"Train accuracy: {accuracy_train * 100:.2f}% ({correct_train} out of {len(pos_dataloader.dataset)})")

	accuracy_test, correct_test = evaluate(supervised_ff, test_dataloader)
	print(f"Test accuracy: {accuracy_test * 100:.2f}% ({correct_test} out of {len(test_dataloader.dataset)})")

	clean_repo()

	print()

######################################################################################
