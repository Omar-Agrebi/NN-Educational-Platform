import numpy as np
from typing import Tuple, Dict, Any
from core.neural_net import NeuralNetwork


def compute_boundary(
    model: NeuralNetwork,
    X: np.ndarray,
    resolution: int = 100,
    margin: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute decision boundary for a 2D binary classifier.
    
    Creates a 100x100 mesh grid over the feature space,
    runs model.forward() over the entire grid,
    and returns contour arrays plus original data points.
    
    Returns dict with: xx, yy, Z, x_points, y_points, labels
    """
    x_min = float(X[:, 0].min()) - margin
    x_max = float(X[:, 0].max()) + margin
    y_min = float(X[:, 1].min()) - margin
    y_max = float(X[:, 1].max()) + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Run forward pass over entire grid in one shot
    probs = model.forward(grid_points)
    Z = probs.reshape(xx.shape)

    return {
        "xx": xx.tolist(),
        "yy": yy.tolist(),
        "Z": Z.tolist(),
        "x_points": X[:, 0].tolist(),
        "y_points": X[:, 1].tolist(),
        "labels": None,  # filled by caller who has y
    }


def compute_boundary_with_labels(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 100,
    margin: float = 0.5,
) -> Dict[str, Any]:
    """Full boundary computation including data point labels."""
    result = compute_boundary(model, X, resolution, margin)
    result["labels"] = y.astype(int).tolist()
    return result
