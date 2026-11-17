from neuron.neuron import Neuron

class ReluNeuron(Neuron):
    def __init__(self, input_count: int) -> None:
        super().__init__(input_count)

    def activation_function(self, z: float) -> float:
        return max(0.0, z)

    def derivative(self, output: float) -> float:
        """Derivative expressed through the neuron's output: 1 if output > 0 else 0."""
        return 1.0 if output > 0 else 0.0