import numpy as np


def softmax_stable(beta: float, fitnesses: np.ndarray) -> np.ndarray:
    """Return numerically stable softmax weights from strategy fitnesses."""
    scaled = beta * (fitnesses - np.max(fitnesses))
    scaled = np.clip(scaled, -500, 0)
    exp_vals = np.exp(scaled)
    total = np.sum(exp_vals)
    if total == 0 or not np.isfinite(total):
        return np.full(len(fitnesses), 1.0 / len(fitnesses))
    return exp_vals / total
