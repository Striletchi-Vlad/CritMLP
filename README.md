# CritMLP
### Get a critically tuned NN specifying only the desired depth(or width) and activation function!  

The network is __automatically built__ in the best way possible regarding the depth/width ratio,
initialization distribution and residual connections to ensure **high performance**, **stability**
and **quick training**.  
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
You can then use the CritMLP as you would any other nn.Module in Pytorch.  

A more advanced example:

```
from crit_mlp import CritMLP

model = CritMLP(in_dim=input_size, out_dim=output_size,
                    depth=20, af='relu-like', neg_slope=0.2, pos_slope=0.5)

```
This will build NN taking inputs of size 14, with an output of size 3 and depth of 10 layers,
with a relu-like activation function with negative slope of 0.2 and positive slope of 0.5.

## Performance Comparison
For a quick test, a network with 20 layers, relu activation function is trained on the scipy wine dataset, for 400 epochs.
Here is how the CritMLP performs against a feed-forward NN with default initialization and one initialized with kaiming normal init (best from pytorch built-ins):

### Gradient norm
Default initialization:
<img src="/assets/no_init_grad_norm.png" alt="no_init" style="height: 200px; width:300px;"/>
