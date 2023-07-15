# CritMLP
### Get a critically tuned NN specifying only the desired depth(or width) and activation function!  

The network is __automatically built__ in the best way possible regarding the depth/width ratio,
initialization distribution and residual connections to ensure **high performance**, **stability**
and **quick training**.  
Currently supported activation functions are ReLU, any ReLU-like function, tanh, and linear. Support for swish and gelu
is underway. The network architecture is, for now, limited to a fully-connected feed-forward network with residual connections.

## Usage

