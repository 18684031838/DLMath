# Convert the example of error backpropagation in mathematics in deep learning to an implementation in the PyTorch framework and output the intermediate variables and gradient calculation data,
# compare with the Excel calculation process, understand the internal principles and internal calculation process of PyTorch implementation
# refer to:https://www.ituring.com.cn/book/2593
import torch
import DLMathBP as bp

# Load input layer data, 64 images, each with 12 neurons
x = bp.load_data()

# Each input neuron is connected to 3 neurons in the hidden layer with different weights and biases. The fixed initialization here is for easy comparison and analysis with Excel in the book. In actual use, parameters are usually initialized randomly using a normal distribution.
w1, b1 = bp.init_hidden_layer1_weight()
h1 = bp.linear(x, w1, b1)
a1 = bp.sigmoid(h1)

w2, b2 = bp.init_hidden_layer2_weight()
h2 = bp.linear(x, w2, b2)
a2 = bp.sigmoid(h2)

w3, b3 = bp.init_hidden_layer3_weight()
h3 = bp.linear(x, w3, b3)
a3 = bp.sigmoid(h3)

# Error of hidden layer neurons
d1 = bp.sigmoid_derivative(a1)
d2 = bp.sigmoid_derivative(a2)
d3 = bp.sigmoid_derivative(a3)

# Output layer result
ow1, ob1 = bp.init_output_layer1_weight()
ow2, ob2 = bp.init_output_layer2_weight()

o = torch.cat((a1.view(64, 1), a2.view(64, 1), a3.view(64, 1)), dim=1)
z1 = bp.linear(o, ow1, ob1)
z2 = bp.linear(o, ow2, ob2)

# Predicted correct variables
o1 = bp.sigmoid(z1)
o2 = bp.sigmoid(z2)

# Derivative of activation function for predicted result
ao1 = bp.sigmoid_derivative(o1)
ao2 = bp.sigmoid_derivative(o2)

# Cost function calculation result
right_answer = bp.load_label()
t1, t2 = right_answer
c = bp.cost(right_answer, (o1, o2))

# Partial derivatives of cost function for output neurons
do1 = o1.view(64, 1) - t1
do2 = o2.view(64, 1) - t2

# Error of neurons in the 3rd layer
delta31 = ao1.view(64, 1) * do1.view(64, 1)
delta32 = ao2.view(64, 1) * do2.view(64, 1)

output_weight = torch.cat((ow1, ow2), dim=0).t()
delta3 = torch.cat((delta31, delta32), dim=1)

sum_weight_delta = (output_weight @ delta3.t()).t()

hidden_layer_derivative = torch.cat((d1.view(64, 1), d2.view(64, 1), d3.view(64, 1)), dim=1)

# Error of neurons in the 2nd layer
delta2 = sum_weight_delta * hidden_layer_derivative

# Partial derivatives of squared error
# Partial derivatives of hidden layer 1
delta21 = delta2[:, 0].view(64, 1)
delta22 = delta2[:, 1].view(64, 1)
delta23 = delta2[:, 2].view(64, 1)
dc_hidden_layer1 = x * delta21.expand(-1, 12)
dc_hidden_layer2 = x * delta22.expand(-1, 12)
dc_hidden_layer3 = x * delta23.expand(-1, 12)