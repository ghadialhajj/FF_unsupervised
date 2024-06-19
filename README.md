# FF_unsupervised

A standalone PyTorch implementation of the Forward-Forward algorithm (specifically the unsupervised example) proposed
by [(Hinton, 2022), Sec 3.2](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ghadialhajj/FF_unsupervised/blob/master/main.ipynb) (make sure to change runtime type to GPU)

File structure:  
.  
├── main.py  
├── main.ipynb  
├── utils.py  
├── MNIST  
├── environment.yml  
├── README.md  
└── LICENSE

The `utils.py` file has the functions to generate the negative examples.

The `prepare_data.py` file downloads the MNIST dataset, generates a dataset of negative images, and saves it as a `.pt`
file that can be loaded directly into the code as a dataset, and then a dataloader. This saves a lot of time compared to
generating negative data on the fly.

The `main.py` file has the `Unsupervised_FF` class along with the training procedure included inside the main block.

Contributions are welcome through pull requests :)
