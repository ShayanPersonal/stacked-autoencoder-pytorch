# stacked-autoencoder-pytorch
Stacked denoising convolutional autoencoder written in Pytorch for some experiments.

This model performs unsupervised reconstruction of the input using a setup similar to Hinton in https://www.cs.toronto.edu/~hinton/science.pdf.
Instead of training layers one at a time, I allow them to train at the same time. Each is trained locally and no backpropagation is used.
The autoencoder is denoising as in http://machinelearning.org/archive/icml2008/papers/592.pdf and convolutional. ReLU activation function is used.

The quality of the feature vector is tested with a linear classifier reinitialized every 10 epochs.

Setup:
- Python 3
- Pytorch 0.3.0 with torchvision

Run "python run.py" to start training.

Observations:
  - Although the loss doesn't propagate through layers the overall quality of the reconstruction improves anyways.
  - When using ReLU, the sparsity of the feature vector at the last layer increases from 50% at the start to 80%+ as you continue training even though sparsity isn't enforced.
  - Augmenting with rotation slows down training and decreases the quality of the feature vector.