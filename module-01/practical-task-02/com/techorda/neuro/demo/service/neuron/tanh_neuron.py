import math
from neuron.neuron import Neuron

class TanhNeuron(Neuron):
    def __init__(self, input_count: int) -> None:
        super().__init__(input_count)

    def activation_function(self, z: float) -> float:
        return math.tanh(z)

    def derivative(self, output: float) -> float:
        return 1 - output**2