# group_model.py
import numpy as np

from subject import Subject


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

    def fit(self) -> np.ndarray:
        """
        Train the group model with algebraic linear regression.
        """
        transformed_sc_stack, transformed_fc_stack = self._preprocess_data()
        return self._algebraic_linear_regression(
            transformed_sc_stack, transformed_fc_stack
        )
