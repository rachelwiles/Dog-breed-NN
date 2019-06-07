# Dog breed neural network :dog2:

## Experimenting with neural networks to classify images of 5 different dog breeds.

### Repo for testing how changing metaparameters and setup of a neural network affects the end accuracy.

_Classes_:
* Poodle
* German Shepherd 
* Dalmatian
* St Bernard
* Pug

_Compatible with_:
* AlexNet
* VGG-19
* ResNet-15
* ResNet-34
* ResNet-152

Generates training & validation graphs over the training period. Creates a confusion matrix of results. 

### Before training, run split.py to split the dataset into the desired training / validation / test split.

_To run from command line_:
* --mod specify which model
* --lr specify the learning rate
* --a specify augmentations (1 for on, 0 for off)
* --s specify the datasplit