from typing import Union
from unittest import TestCase

import numpy as np
import torch

from bbopy.problems.base import SingleObjectiveProblem


def _get_prob(problem, backend, **kwargs):
    return problem(backend=backend, **kwargs)


class DummyProblem(SingleObjectiveProblem):
    dim = 2
    n_obj = 1
    n_constr = 0
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]

    def _evaluate(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()
        return back.sum(x ** 2, **kwargs).reshape(-1, 1)


class TestSingleObjectiveProblem(TestCase):
    problem = DummyProblem
    problem_kwargs = {}

    def _test_backend(self, backend, dtypes):
        prob = _get_prob(self.problem, backend=backend, **self.problem_kwargs)
        for dtype in dtypes:
            for optimizer in prob._optimizers:
                if backend == "torch":
                    x = torch.tensor(optimizer, dtype=dtype)
                else:
                    x = np.array(optimizer, dtype=dtype)
                y = prob(x)
                if backend == "torch":
                    self.assertIsInstance(y, torch.Tensor)
                else:
                    self.assertIsInstance(y, np.ndarray)
                self.assertEqual(y.dtype, dtype)
                self.assertAlmostEqual(prob._optimal_value, y[0, 0].item(), places=5)

    def test_optimizer(self):
        # Test torch backend
        self._test_backend(
            backend="torch",
            dtypes=[torch.float32, torch.double],
        )
        # Test numpy backend
        self._test_backend(
            backend="numpy",
            dtypes=[np.float64],
        )
        # Test with a wrong backend
        with self.assertRaises(NotImplementedError):
            self._test_backend(
                backend="test",
                dtypes=[np.float64],
            )
