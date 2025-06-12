"""
Contains the functions to train the model.
"""
from collections.abc import Callable
from typing import Any
import math
import cupy as cp
import numpy as np
from cuml.linear_model import Lasso


def calc_alpha_max(X: cp.ndarray, y: cp.ndarray) -> float:
    """
    Computes alpha_max = max_{col in X.T}(dot(col, y)) / len(y)
    in a single GPU-accelerated matrix multiplication.

    Equivalent to the old loop:
        inner_products = []
        for row in X.T:
            inner_products.append(cp.dot(row, y))
        alpha_max = max(inner_products) / N
    """
    N = len(y)
    # Compute the inner products in a single matrix multiplication
    products = X.T @ y

    # Grab the maximum and divide by N
    alpha_max = cp.max(products) / N
    return alpha_max

def calc_alpha_grid(X: cp.ndarray, y: cp.ndarray, num_alphas: int = 100) -> list[float]:
    """
    Computes a grid of alpha values between 0 and alpha_max.
    """
    alpha_max = calc_alpha_max(X, y)
    alpha_min = 0.00005 * alpha_max
    alpha_max = 100 * alpha_min
    return cp.logspace(math.log(alpha_min), math.log(alpha_max), num=num_alphas, base=math.exp(1))

def calc_next_alpha(alpha1: float, alpha2: float) -> tuple[float, float]:
    """
    Calculates the midpoint between two alpha values, on natural log scale.
    """
    alpha1 = math.log(alpha1)
    alpha2 = math.log(alpha2)
    alpha_next = (alpha1 + alpha2) / 2
    return math.exp(alpha_next)

def lasso_regression(X: cp.ndarray, y: cp.ndarray, alpha: float, tol: float=0.000000000000000001, max_iter: int=200) -> cp.ndarray:
    """
    Fits a Lasso regression model using scikit-learn's Lasso function.
    """
    # Fit a Lasso model using scikit-learn's Lasso function
    lasso_model = Lasso(alpha=alpha, fit_intercept=False, tol=tol, max_iter=max_iter)
    lasso_model.fit(X, y)

    # Return the fitted coefficients
    return lasso_model.coef_

def binary_search_train(X: cp.ndarray, y: cp.ndarray, max_iter=10) -> tuple[cp.ndarray, float]:
    """
    Performs a binary search to find the optimal alpha value for Lasso regression,
    which results in ~80% nonzero rule coefficients.
    """
    #small_alpha = calc_alpha_max(X, y) * 0.00001#small_alpha = calc_alpha_max(X, y) * 0.00015
    #big_alpha = small_alpha * 10#big_alpha = small_alpha * 4
    small_alpha = calc_alpha_max(X, y) * 0.00004
    big_alpha = small_alpha * 10

    for _ in range(max_iter):
        mid_alpha = calc_next_alpha(small_alpha, big_alpha)
        rules = lasso_regression(X, y, mid_alpha)
        rules_density = cp.count_nonzero(rules) / len(rules)

        print(f"Alpha: {mid_alpha}, Density: {rules_density}")
        print(f"Max alpha: {big_alpha}, min alpha: {small_alpha}")

        if abs(rules_density - 0.8) < 0.01:
            break
        if rules_density >= 0.8:
            small_alpha = mid_alpha
        else:
            big_alpha = mid_alpha
    else:
        print("Warning: Binary search did not converge.")

    return rules, mid_alpha

class Subject:
    """
    Represents a single subject.
    """
    def __init__(self, subject_id: int, sc: cp.ndarray, fc: cp.ndarray, transform: Callable[[cp.ndarray, cp.ndarray], cp.ndarray]) -> None:
        self.subject_id = subject_id
        self.transformed_sc, self.transformed_fc = transform(sc, fc)

    def calc_predicted_fc(self, rules: cp.ndarray) -> cp.ndarray:
        """
        Make prediction for the subject with the given rules.
        """
        return self.sc @ rules @ self.sc

class GroupLevelModel:
    """
    Represents a group of subjects.
    """
    def __init__(self, subjects: list[Subject]) -> None:
        self.subjects = subjects

    def _algebraic_linear_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fits a linear regression model using algebraic linear regression.
        """
        return np.linalg.pinv(X) @ y

    def _preprocess_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for training.
        """
        transformed_scs = [subject.transformed_sc for subject in self.subjects]
        transformed_fcs = [subject.transformed_fc for subject in self.subjects]
        
        transformed_sc_stack = np.vstack(transformed_scs)
        transformed_fc_stack = np.hstack(transformed_fcs)
        return transformed_sc_stack, transformed_fc_stack

    def fit(self, algebraic=False) -> np.ndarray:
        """
        Train the group model with algebraic linear regression.
        """
        transformed_sc_stack, transformed_fc_stack = self._preprocess_data()
        return self._algebraic_linear_regression(transformed_sc_stack, transformed_fc_stack)