![alt text](predify_logo.png)
### Enables you to easily extend your deep neural network with predictive coding dynamics.
Link to the preprint: https://arxiv.org/abs/2106.02749

Tutorial: [predify-demo.ipynb](https://github.com/miladmozafari/predify/blob/master/examples/predify_demo.ipynb) ([Run in Google Colab](https://colab.research.google.com/github/miladmozafari/predify/blob/master/examples/predify_demo.ipynb))
# Installation

To setup this package, you can install Anaconda or Miniconda
```
# Clone the repository
git clone https://github.com/miladmozafari/predify

# Create a new conda environment
conda create -n predifyproject python=3
conda activate predifyproject

# Install all the dependencies
cd predify
pip install -r requirements.txt
```
```
# Alternatively, one can just run the following command
pip install git+https://github.com/miladmozafari/predify.git
```


You can also set it up as a package in a development version

```
python setup.py develop
```

You can now `import predify` from anywhere as long as you are in the conda environment `predifyproject`

# Getting Started
Using the following template, you can generate Predified version of your deep network:
```python
from predify import predify

net = # load your network
predify(net, 'address_to_the_config_file.toml')
```

Where the config file contains the following (here we show predictive VGG16 as an example):
```toml
# network name (default: "Network")
name = "PVGG16"

# imports for prediction modules (mandatory if custom predictor modules are defined)
imports = [
"from torch.nn import Sequential, ReLU, ConvTranspose2d",
]

# indicates utilization of automatic gradient scaling (default: false)
gradient_scaling = true

# to use shared or separate hyperparameters for PCoders (default: false)
shared_hyperparameters = false

# input size [channels, height, width] (mandatory)
input_size = [3, 224, 224]

# pcoders (defining [[pcoders]] for each PCoder is mandatory). Order is important.
# module (mandatory): pytorch name of each module in the target network to be converted into an encoder
# predictor (optional): the pytorch module for generating predictions. By default, it will be upsample+conv_transpose)
# hyperparameters (optional): default value is {feedforward=0.3, feedback=0.3, pc=0.01}. If shared_hyperparameters=true, the values of the first PCoder will be used for all of them.

[[pcoders]]
module = "features[3]"     # so the target module is vgg16.features[3]
predictor = "ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))"
hyperparameters = {feedforward=0.2, feedback=0.05, pc=0.01}

[[pcoders]]
module = "features[8]"
predictor = "Sequential(ConvTranspose2d(128, 64, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)), ReLU(inplace=True))"
hyperparameters = {feedforward=0.4, feedback=0.1, pc=0.01}

[[pcoders]]
module = "features[15]"
predictor = "Sequential(ConvTranspose2d(256, 128, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))"
hyperparameters = {feedforward=0.4, feedback=0.1, pc=0.01}

[[pcoders]]
module = "features[22]"
predictor = "Sequential(ConvTranspose2d(512, 256, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))"
hyperparameters = {feedforward=0.5, feedback=0.1, pc=0.01}

[[pcoders]]
module = "features[29]"
predictor = "Sequential(ConvTranspose2d(512, 512, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))"
hyperparameters = {feedforward=0.6, feedback=0.0, pc=0.01}
```
