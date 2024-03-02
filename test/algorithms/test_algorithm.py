from unittest import TestCase

import torch

from bbopy.algorithms import BoTorchAlgorithm
from test.problems.test_problem import DummyProblem


class DummyBoTorchAlgorithm(BoTorchAlgorithm):
    def ask(self):
        if self._train_x is None:
            self._train_x = torch.randn((self.n_init, len(self.bounds)))
            return self._train_x
        new_x = torch.randn((1, len(self.bounds)))
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y) -> None:
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])


class TestBoTorchAlgorithm(TestCase):
    algorithm_class = DummyBoTorchAlgorithm
    algorithm_kwargs = {"n_init": 5}
    problem = DummyProblem(backend="torch", maximize=True)

    def _get_algorithm(self):
        algorithm = self.algorithm_class(**self.algorithm_kwargs)
        algorithm.setup(self.problem)
        return algorithm

    def test_ask_and_tell(self):
        algorithm = self._get_algorithm()
        # First iteration should initialize the tensor
        train_x = algorithm.ask()
        self.assertIsInstance(train_x, torch.Tensor)
        self.assertEqual(train_x.shape[0], algorithm.n_init)
        self.assertEqual(train_x.shape[1], len(self.problem.bounds))
        train_y = self.problem(train_x)
        self.assertIsInstance(train_y, torch.Tensor)
        self.assertEqual(train_y.shape[0], algorithm.n_init)
        algorithm.tell(train_y)
        # Second iteration should add a new point
        new_x = algorithm.ask()
        self.assertIsInstance(new_x, torch.Tensor)
        self.assertEqual(new_x.shape[0], 1)
        self.assertEqual(new_x.shape[1], len(self.problem.bounds))
        new_y = self.problem(new_x)
        self.assertIsInstance(new_y, torch.Tensor)
        self.assertEqual(new_y.shape[0], 1)
        algorithm.tell(new_y)
        # train_x and train_y should containt n_init + 1 points
        self.assertEqual(algorithm.train_x.shape[0], algorithm.n_init + 1)
        self.assertEqual(algorithm.train_y.shape[0], algorithm.n_init + 1)
