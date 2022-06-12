/* GAN using Go. */

package main

import (
	"fmt"
	"math/rand"
	"time"
)

const (
	// Number of neurons in the input layer.
	inputNeurons = 2
	// Number of neurons in the hidden layer.
	hiddenNeurons = 2
	// Number of neurons in the output layer.
	outputNeurons = 1
	// Number of training iterations.
	iterations = 10000
	// Learning rate.
	learningRate = 0.1
)

// A neuron.
type Neuron struct {
	// The weights of the neuron.
	weights []float64
	// The bias of the neuron.
	bias float64
}

// A layer of neurons.
type Layer struct {
	// The neurons in the layer.
	neurons []Neuron
}

// A neural network.
type NeuralNetwork struct {
	// The input layer.
	inputLayer Layer
	// The hidden layer.
	hiddenLayer Layer
	// The output layer.
	outputLayer Layer
}

// Initialize the neural network.
func (nn *NeuralNetwork) initialize() {
	// Initialize the input layer.
	nn.inputLayer.neurons = make([]Neuron, inputNeurons)
	for i := 0; i < inputNeurons; i++ {
		nn.inputLayer.neurons[i].weights = make([]float64, hiddenNeurons)
		for j := 0; j < hiddenNeurons; j++ {
			nn.inputLayer.neurons[i].weights[j] = rand.Float64()
		}
		nn.inputLayer.neurons[i].bias = rand.Float64()
	}

	// Initialize the hidden layer.
	nn.hiddenLayer.neurons = make([]Neuron, hiddenNeurons)
	for i := 0; i < hiddenNeurons; i++ {
		nn.hiddenLayer.neurons[i].weights = make([]float64, outputNeurons)
		for j := 0; j < outputNeurons; j++ {
			nn.hiddenLayer.neurons[i].weights[j] = rand.Float64()
		}
		nn.hiddenLayer.neurons[i].bias = rand.Float64()
	}

	// Initialize the output layer.
	nn.outputLayer.neurons = make([]Neuron, outputNeurons)
	for i := 0; i < outputNeurons; i++ {
		nn.outputLayer.neurons[i].weights = make([]float64, 0)
		nn.outputLayer.neurons[i].bias = rand.Float64()
	}
}

// Feed forward the neural network.
func (nn *NeuralNetwork) feedForward(inputs []float64) []float64 {
	// Feed forward the input layer.
	for i := 0; i < inputNeurons; i++ {
		nn.inputLayer.neurons[i].bias = inputs[i]
	}

	// Feed forward the hidden layer.
	for i := 0; i < hiddenNeurons; i++ {
		var sum float64
		for j := 0; j < inputNeurons; j++ {
			sum += nn.inputLayer.neurons[j].bias * nn.inputLayer.neurons[j].weights[i]
		}
		nn.hiddenLayer.neurons[i].bias = sigmoid(sum)
	}

	// Feed forward the output layer.
	for i := 0; i < outputNeurons; i++ {
		var sum float64
		for j := 0; j < hiddenNeurons; j++ {
			sum += nn.hiddenLayer.neurons[j].bias * nn.hiddenLayer.neurons[j].weights[i]
		}
		nn.outputLayer.neurons[i].bias = sigmoid(sum)
	}

	// Return the output.
	outputs := make([]float64, outputNeurons)
	for i := 0; i < outputNeurons; i++ {
		outputs[i] = nn.outputLayer.neurons[i].bias
	}
	return outputs
}

// Train the neural network.
func (nn *NeuralNetwork) train(inputs []float64, targets []float64) {
	// Feed forward the neural network.
	outputs := nn.feedForward(inputs)

	// Calculate the output layer error.
	outputLayerErrors := make([]float64, outputNeurons)
	for i := 0; i < outputNeurons; i++ {
		outputLayerErrors[i] = targets[i] - outputs[i]
	}

	// Calculate the hidden layer error.
	hiddenLayerErrors := make([]float64, hiddenNeurons)
	for i := 0; i < hiddenNeurons; i++ {
		var sum float64
		for j := 0; j < outputNeurons; j++ {
			sum += outputLayerErrors[j] * nn.hiddenLayer.neurons[i].weights[j]
		}
		hiddenLayerErrors[i] = sum
	}

	// Update the output layer weights.
	for i := 0; i < outputNeurons; i++ {
		for j := 0; j < hiddenNeurons; j++ {
			nn.hiddenLayer.neurons[j].weights[i] += learningRate * outputLayerErrors[i] * nn.hiddenLayer.neurons[j].bias
		}
	}

	// Update the hidden layer weights.
	for i := 0; i < hiddenNeurons; i++ {
		for j := 0; j < inputNeurons; j++ {
			nn.inputLayer.neurons[j].weights[i] += learningRate * hiddenLayerErrors[i] * nn.inputLayer.neurons[j].bias
		}
	}
}

// Sigmoid function.
func sigmoid(x float64) float64 {
	return 1 / (1 + (0 - x))
}

// Main function.
func main() {
	// Initialize the random number generator.
	rand.Seed(time.Now().UTC().UnixNano())

	// Create the neural network.
	var nn NeuralNetwork
	nn.initialize()

	// Train the neural network.
	for i := 0; i < iterations; i++ {
		// Create the inputs.
		inputs := make([]float64, inputNeurons)
		for j := 0; j < inputNeurons; j++ {
			inputs[j] = rand.Float64()
		}

		// Create the targets.
		targets := make([]float64, outputNeurons)
		for j := 0; j < outputNeurons; j++ {
			targets[j] = inputs[0] * inputs[1]
		}

		// Train the neural network.
		nn.train(inputs, targets)
	}

	// Test the neural network.
	inputs := []float64{0.5, 0.5}
	outputs := nn.feedForward(inputs)
	fmt.Println(outputs)
}
