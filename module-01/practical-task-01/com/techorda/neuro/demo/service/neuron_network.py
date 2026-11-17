import math
import random
from typing import List
from neuron_layer import NeuronLayer
from neuron.tanh_neuron import TanhNeuron
from neuron.sigmoid_neuron import SigmoidNeuron
from data_generator import generate_data
from config import (COORD_RANGE, N_TRAIN, N_VAL, TRAIN_SEED, VAL_SEED, HIDDEN_SIZE, EPOCHS,
                    LEARNING_RATE, LR_DECAY, BATCH_SIZE, PATIENCE, CHECK_EVERY)

class NeuronNetwork:
    def __init__(self, hidden_size: int = HIDDEN_SIZE, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE, batch_size: int = BATCH_SIZE):
        self.layers: List[NeuronLayer] = []

        """
        Implement this constructor
        to create a neural network with a specified architecture.
        Your network should train during initialization
        """
        self.layers.append(NeuronLayer([TanhNeuron(2) for _ in range(hidden_size)]))
        self.layers.append(NeuronLayer([SigmoidNeuron(hidden_size) for _ in range(1)]))
        for layer in self.layers:
            self.initialize_weights(layer)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.history = {"train_loss": [], "val_loss": []}

        X_train, y_train = generate_data(N_TRAIN, seed=TRAIN_SEED)
        X_val, y_val = generate_data(N_VAL, seed=VAL_SEED)
        self.train(training_inputs=X_train, training_outputs=y_train, val_inputs=X_val, val_outputs=y_val)

    def initialize_weights(self, layer: NeuronLayer) -> None:
        for neuron in layer.neurons:
            neuron.weights = [random.uniform(-0.5, 0.5) for _ in neuron.weights]
            neuron.bias = random.uniform(-0.5, 0.5)

    def _normalize(self, inputs: List[float]) -> List[float]:
        return [value / COORD_RANGE for value in inputs]

    def predict(self, inputs: List[float]) -> List[float]:
        output = self._normalize(inputs)
        for layer in self.layers:
            output = layer.activate(output)
        return output

    """ implement this method
    to train the neural network using backpropagation
    for the given number of epochs and learning rate.
    Use Mean Squared Error (MSE) as the loss function.
    Log the total error every 1000 epochs.
    trainingInputs - input data for training
    trainingOutputs - expected output data for training
    epochs - number of training iterations
    learningRate - step size for weight updates
    Hint: You will need to implement forward propagation,
    error calculation, backpropagation, and weight updates.
    """
    # Loss: Binary Cross-Entropy. Previously MSE (as in the template docstring above),
    # changed to BCE after the Git AI review.
    def train(self, training_inputs=None, training_outputs=None, val_inputs=None, val_outputs=None, epochs: int = None, learning_rate: float = None, batch_size: int = None, patience: int = PATIENCE) -> None:
        if training_inputs is None or training_outputs is None:
            training_inputs, training_outputs = generate_data(N_TRAIN, seed=TRAIN_SEED)
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if batch_size is None:
            batch_size = self.batch_size

        hidden_layer, output_layer = self.layers
        indices = list(range(len(training_inputs)))
        best_val_loss = float("inf")
        best_weights = None
        no_improvement = 0

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                # gradients for each neuron: one value per weight + one for the bias (last)
                output_grads = [[0.0] * (len(neuron.weights) + 1) for neuron in output_layer.neurons]
                hidden_grads = [[0.0] * (len(neuron.weights) + 1) for neuron in hidden_layer.neurons]

                for idx in batch:
                    inputs = self._normalize(training_inputs[idx])
                    expected = training_outputs[idx]
                    hidden_outputs = hidden_layer.activate(inputs)
                    outputs = output_layer.activate(hidden_outputs)

                    # BCE + sigmoid: the delta of the output neuron is simply (expected - output)
                    output_deltas = []
                    for k in range(len(outputs)):
                        total_loss += self._bce(expected[k], outputs[k])
                        output_deltas.append(expected[k] - outputs[k])

                    hidden_deltas = []
                    for j, neuron in enumerate(hidden_layer.neurons):
                        downstream = 0.0
                        for k, out_neuron in enumerate(output_layer.neurons):
                            downstream += out_neuron.weights[j] * output_deltas[k]
                        hidden_deltas.append(downstream * neuron.derivative(hidden_outputs[j]))

                    for k, delta in enumerate(output_deltas):
                        for i in range(len(hidden_outputs)):
                            output_grads[k][i] += delta * hidden_outputs[i]
                        output_grads[k][-1] += delta

                    for j, delta in enumerate(hidden_deltas):
                        for i in range(len(inputs)):
                            hidden_grads[j][i] += delta * inputs[i]
                        hidden_grads[j][-1] += delta

                step = learning_rate / len(batch)
                self._update_layer(output_layer, output_grads, step)
                self._update_layer(hidden_layer, hidden_grads, step)

            train_loss = total_loss / len(training_inputs)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: BCE = {train_loss:.6f}")

            if epoch % CHECK_EVERY == 0:
                learning_rate *= LR_DECAY
                self.history["train_loss"].append(train_loss)

                if val_inputs is not None:
                    val_loss = self.loss(val_inputs, val_outputs)
                    self.history["val_loss"].append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = [[(neuron.weights[:], neuron.bias) for neuron in layer.neurons] for layer in self.layers]
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

        if best_weights is not None:
            for layer, saved_layer in zip(self.layers, best_weights):
                for neuron, (weights, bias) in zip(layer.neurons, saved_layer):
                    neuron.weights = weights
                    neuron.bias = bias

    def _update_layer(self, layer: NeuronLayer, grads, step: float) -> None:
        for neuron, neuron_grads in zip(layer.neurons, grads):
            for i in range(len(neuron.weights)):
                neuron.weights[i] += step * neuron_grads[i]
            neuron.bias += step * neuron_grads[-1]

    def _bce(self, expected: float, predicted: float) -> float:
        predicted = min(max(predicted, 1e-10), 1 - 1e-10)
        return -(expected * math.log(predicted) + (1 - expected) * math.log(1 - predicted))

    def loss(self, inputs, outputs) -> float:
        total_loss = 0.0
        for x, expected in zip(inputs, outputs):
            total_loss += self._bce(expected[0], self.predict(x)[0])
        return total_loss / len(inputs)

    def classify(self, x: float, y: float, threshold: float = 0.5) -> int:
        return 1 if self.predict([x, y])[0] >= threshold else 0

    def evaluate(self, test_inputs, test_outputs) -> float:
        correct = 0
        for (x, y), expected in zip(test_inputs, test_outputs):
            if self.classify(x, y) == int(expected[0]):
                correct += 1
        return correct / len(test_inputs)
