# subject.py
from collections.abc import Callable
import cupy as cp


class Subject:
    """
    Represents a single subject.
    """

    def __init__(
        self,
        subject_id: int,
        sc: cp.ndarray,
        fc: cp.ndarray,
        transform: Callable[[cp.ndarray, cp.ndarray], cp.ndarray],
    ) -> None:
        self.subject_id = subject_id
        self.transformed_sc, self.transformed_fc = transform(sc, fc)

    def calc_predicted_fc(self, rules: cp.ndarray) -> cp.ndarray:
        """
        Make prediction for the subject with the given rules.
        """
        return self.transformed_sc @ rules @ self.transformed_sc
