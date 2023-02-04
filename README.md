# FF_unsupervised

An implementation of unsupervised example of the Forward-Forward algorithm proposed
by [(Hinton, 2020), Sec 3.2](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

.  
├── main.py  
├── prepare_data.py  
└── utils.py

The `utils.py` file has the functions to generate the negative examples.

The `prepare_data.py` file downloads the MNIST dataset, generates a dataset of negative images, and saves it as a `.pt`
file that can be loaded directly into the code as a dataset, and then a dataloader. This saves a lot of time compared to
generating negative data on the fly.

The `main.py` file has the `Unsupervised_FF` class along with the training procedure included inside the main block.