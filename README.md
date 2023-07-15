# CritMLP
### Get a critically tuned NN specifying only the desired depth(or width) and activation function!  

This project is an aplication of the theoretical concepts discussed in [Deep Learning Theory](https://deeplearningtheory.com/).

The network is __automatically built__ in the best way possible regarding the depth/width ratio,
initialization distribution and residual connections to ensure **high performance**, **stability**
and **quick training**.  

This is a side effect of **criticality**, a state describing neural networks with outputs belonging to **a Gaussian distribution with variance = 1**, which can be achieved by enforcing certain design principles at creation.

Currently supported activation functions are ReLU, any ReLU-like function, tanh, and linear. Support for swish and gelu
is underway. The network architecture is, for now, limited to a fully-connected feed-forward network with residual connections.

## Usage
For a quick demo, clone the repository and run
```
python demo.py
```
Clone the repository or manually download crit_functions.py and crit_mlp.py. Afterwards, just import CritMLP and instantiate it as such:
```
from crit_mlp import CritMLP

model = CritMLP(in_dim, out_dim, depth, 'relu')

```
You can then use the CritMLP just as you would any other nn.Module in Pytorch.  

A more advanced example:

```
from crit_mlp import CritMLP

model = CritMLP(in_dim=14, out_dim=3,
                    depth=20, af='relu-like', neg_slope=0.2, pos_slope=0.5)

```
This will build NN taking inputs of size 14, with an output of size 3 and depth of 20 layers,
with a relu-like activation function with negative slope of 0.2 and positive slope of 0.5.

## Performance Comparison
For a quick test, a network with 20 layers, ReLU activation function is trained on the scipy wine dataset, for 400 epochs.
Here is how the CritMLP performs against a feed-forward NN with default initialization and one initialized with kaiming normal init (best from pytorch built-ins):

### Accuracy
#### Default initialization:
<img src="/assets/no_init_acc.png" alt="default" style="height: 500px; width:600px;"/>

#### Kaiming normal initialization:
<img src="/assets/kaiming_normal_acc.png" alt="kaiming_normal" style="height: 500px; width:600px;"/>

#### Critical initialization (CritMLP):
<img src="/assets/crit_init_acc.png" alt="critical" style="height: 500px; width:600px;"/>

### Loss
#### Default initialization:
<img src="/assets/no_init_loss.png" alt="default" style="height: 500px; width:600px;"/>

#### Kaiming normal initialization:
<img src="/assets/kaiming_normal_loss.png" alt="kaiming_normal" style="height: 500px; width:600px;"/>

#### Critical initialization (CritMLP):
<img src="/assets/crit_init_loss.png" alt="critical" style="height: 500px; width:600px;"/>

### Gradient norm
#### Default initialization:
<img src="/assets/no_init_grad_norm.png" alt="default" style="height: 500px; width:600px;"/>

#### Kaiming normal initialization:
<img src="/assets/kaiming_normal_grad_norm.png" alt="kaiming_normal" style="height: 500px; width:600px;"/>

#### Critical initialization (CritMLP):
<img src="/assets/crit_init_grad_norm.png" alt="critical" style="height: 500px; width:600px;"/>
