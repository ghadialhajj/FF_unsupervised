import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from torch import tensor, Tensor


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

    blur_filter_1 = np.array(((0, 0, 0), (0.25, 0.5, 0.25), (0, 0, 0)))
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

    assert image_1.shape == image_2.shape, "Incompatible images and mask shapes."

    mask = create_mask((image_1.shape[0], image_1.shape[1]))

    image_1 = torch.mul(image_1, mask)
    image_2 = torch.mul(image_2, 1-mask)

    return torch.add(image_1, image_2)


def create_negative_batch(images: Tensor):
    neg_imgs = []
    batch_size = images.shape[0]
    for _ in range(batch_size):
        idx1, idx2 = np.random.randint(batch_size, size=2)
        neg_imgs.append(create_negative_image(images[idx1].squeeze(), images[idx2].squeeze()))
    return torch.unsqueeze(torch.stack(neg_imgs), dim=1)


if __name__ == '__main__':
    import torchvision

    # Load the MNIST dataset
    mnist = torchvision.datasets.MNIST(root='data/', download=True)

    # Get the first instance of the digit 1
    image_1, _ = mnist[np.random.randint(len(mnist))]
    image_2, _ = mnist[np.random.randint(len(mnist))]

    image_1 = torch.as_tensor(np.asarray(image_1))
    image_2 = torch.as_tensor(np.asarray(image_2))

    mask = create_mask((28, 28))
    image = create_negative_image(image_1, image_2)

    plt.figure()

    # Create the subplot
    fig, ax = plt.subplots(1, 5)
    images = [image_1, mask, image, 1-mask, image_2]
    names = ["image_1", "mask", "image", "1-mask", "image_2"]
    # Add the images to the subplot
    for i, image in enumerate(images):
        ax[i].imshow(image, cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(names[i], y=-0.5)

    # Show the subplot
    plt.tight_layout()
    plt.show()
