import random
from typing import List, Tuple
from config import VALUE_RANGE

def max_of_three(x1: float, x2: float, x3: float) -> float:
    return max(x1, x2, x3)

def generate_data(n_samples: int, seed: int = 42) -> Tuple[List[List[float]], List[List[float]]]:
    rng = random.Random(seed)
    inputs = []
    outputs = []
    for _ in range(n_samples):
        x1 = rng.uniform(-VALUE_RANGE, VALUE_RANGE)
        x2 = rng.uniform(-VALUE_RANGE, VALUE_RANGE)
        x3 = rng.uniform(-VALUE_RANGE, VALUE_RANGE)
        inputs.append([x1, x2, x3])
        outputs.append([max_of_three(x1, x2, x3)])

    return inputs, outputs
