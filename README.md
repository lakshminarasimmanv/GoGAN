# GAN in Go

This is a GAN written in Go. It is a very simple GAN that only has two input neurons, two hidden neurons, and one output neuron. The training data is generated randomly.

The GAN is trained for 10000 iterations. The learning rate is set to 0.1.

## Usage

To use the GAN, first create a new `NeuralNetwork`:

```go
var nn NeuralNetwork
```

Then, initialize the neural network:

```go
nn.initialize()
```

To train the neural network, use the `train` function:

```go
nn.train(inputs, targets)
```

Where `inputs` is a slice of floats containing the input values, and `targets` is a slice of floats containing the target values.

To test the neural network, use the `feedForward` function:

```go
outputs := nn.feedForward(inputs)
```

Where `inputs` is a slice of floats containing the input values, and `outputs` is a slice of floats containing the output values.
