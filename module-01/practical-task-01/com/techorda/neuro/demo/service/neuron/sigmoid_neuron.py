import math
from neuron.neuron import Neuron

class SigmoidNeuron(Neuron):
    def __init__(self, input_count: int) -> None:
        super().__init__(input_count)

    def activation_function(self, z: float) -> float:
        z = max(-500.0, min(500.0, z))
        return 1 / (1 + math.exp(-z))

    def derivative(self, output: float) -> float:
        return output * (1.0 - output)