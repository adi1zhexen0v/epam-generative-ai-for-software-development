import random
from typing import List, Tuple
from config import RADIUS, COORD_RANGE

def is_inside_circle(x: float, y: float) -> float:
    hypotenuse = x**2 + y**2
    return 1.0 if hypotenuse <= RADIUS**2 else 0.0

def generate_data(n_samples: int, seed: int = 42) -> Tuple[List[List[float]], List[List[float]]]:
    rng = random.Random(seed)
    inputs = []
    outputs = []
    for _ in range(n_samples):
        x = rng.uniform(-COORD_RANGE, COORD_RANGE)
        y = rng.uniform(-COORD_RANGE, COORD_RANGE)
        inputs.append([x, y])
        outputs.append([is_inside_circle(x, y)])

    return inputs, outputs
