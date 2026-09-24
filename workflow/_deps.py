"""Shared imports for the refactored Paper 3 analysis modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS
from scipy.stats import t as student_t
from scipy.stats import f as f_dist


if __name__ == "__main__":
    print("Paper 3 dependencies imported successfully.")
