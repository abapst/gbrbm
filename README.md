# GBRBM

Demo code for a Gaussian-Bernoulli Restricted Boltzmann Machine that
can train on highly sparse data using a collaborative filtering scheme and reconstruct training examples with missing values.

![gbrbm](https://user-images.githubusercontent.com/12631256/50627549-c7468580-0ee8-11e9-9ea3-d2c784a4b2bf.png)

Requirements:
  - python 2.7
  - theano + Nvidia GPU with CUDA
  - matplotlib
  - numpy
  - Pillow (or PIL)

## To install and run the CIFAR-10 demo:

Clone the repo:
```
git clone https://github.com/abapst/gbrbm.git
```

Download the CIFAR-10 data:
```
./get_cifar_data.sh
```

Run the demo:
```
make cifar
```
This will train a GBRBM on the CIFAR-10 "car" class with 10% of the pixels missing from each training image (spatial_sparsity=0.9) for 100 epochs. The model will be stored as a pickled dict in the models/ directory. The top 100 filters with the largest L2-norms will also be plotted at every epoch and saved as .png images in rbm_plots/. At the end, the model will be tested on 100 test examples and the reconstructions along with their ground truth will be stored in rbm_plots/ as well.
