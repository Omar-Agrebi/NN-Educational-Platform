import numpy as np
from typing import Tuple
from core.neural_net import NeuralNetwork
from models.responses import BoundaryData


def compute_boundary(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 80,
    margin: float = 0.5,
) -> BoundaryData:
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid, training=False).reshape(xx.shape)

    return BoundaryData(
        xx=xx.tolist(),
        yy=yy.tolist(),
        Z=Z.tolist(),
        x_points=X[:, 0].tolist(),
        y_points=X[:, 1].tolist(),
        labels=y.astype(int).tolist(),
    )
