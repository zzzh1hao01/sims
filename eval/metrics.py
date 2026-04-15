"""
Evaluation metrics for SubPOP distribution prediction.

Primary metric: Wasserstein Distance (Earth Mover's Distance).
  - Accounts for ordinal structure of answer options (e.g., Likert scale).
  - Penalizes larger deviations more than one-hot accuracy would.
  - Lower is better.

Baselines provided for reference:
  - uniform_wasserstein: uniform distribution over options.
  - majority_wasserstein: all mass on the most common human choice.
"""

import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein(
    predicted_probs: np.ndarray,
    target_probs: np.ndarray,
    ordinals: np.ndarray,
) -> float:
    """
    Wasserstein distance between predicted and human distributions.

    scipy.stats.wasserstein_distance treats ordinals as the support
    of a 1-D discrete measure, so ordinal spacing is respected.

    Args:
        predicted_probs: [num_options] predicted probability mass
        target_probs:    [num_options] human response distribution
        ordinals:        [num_options] ordinal scale positions (e.g. 1,2,3,4,5)
    """
    return wasserstein_distance(
        u_values=ordinals,
        v_values=ordinals,
        u_weights=predicted_probs,
        v_weights=target_probs,
    )


def uniform_wasserstein(num_options: int, ordinals: np.ndarray, target_probs: np.ndarray) -> float:
    uniform = np.ones(num_options) / num_options
    return compute_wasserstein(uniform, target_probs, ordinals)


def majority_wasserstein(ordinals: np.ndarray, target_probs: np.ndarray) -> float:
    majority = np.zeros_like(target_probs)
    majority[np.argmax(target_probs)] = 1.0
    return compute_wasserstein(majority, target_probs, ordinals)
