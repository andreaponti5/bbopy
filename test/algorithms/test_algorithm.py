from typing import Type, Dict, Union, List
from unittest import TestCase

import numpy as np
import torch
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem as PymooProblem
from pymoo.problems.static import StaticProblem

from bbopy.algorithms import BoTorchAlgorithm, PymooAlgorithm, Algorithm
from bbopy.problems import Problem
from test.problems.test_problem import DummyProblem


class DummyBoTorchAlgorithm(BoTorchAlgorithm):
    def ask(self):
        if self._train_x is None:
            self._train_x = torch.randn((self.n_init, len(self.bounds)))
            return self._train_x
        new_x = torch.randn((1, len(self.bounds)))
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y):
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])


class DummyPymooAlgorithm(PymooAlgorithm):
    def setup(
            self,
            problem: Problem,
    ) -> None:
        self._settings = PymooProblem(n_var=problem.dim,
                                      n_obj=problem.n_obj,
                                      n_constr=problem.n_constr,
                                      xl=np.array(problem.bounds)[:, 0],
                                      xu=np.array(problem.bounds)[:, 1])

    def ask(self):
        pop = Population.empty()
        pop_x = np.random.rand(self.pop_size, self._settings.n_var)
        self._curr_pop = pop.new("X", np.atleast_2d(pop_x))
        return self._curr_pop.get("X")

    def tell(self, y):
        self._n_eval += len(y)
        static = StaticProblem(self._settings, F=y)
        Evaluator().eval(static, self._curr_pop)


class TestAlgorithm(TestCase):
    algorithm_class: Union[Type[Algorithm], Type[BoTorchAlgorithm], Type[PymooAlgorithm]]
    algorithm_kwargs: Dict
    problem: Problem

    def _get_algorithm(self):
        algorithm = self.algorithm_class(**self.algorithm_kwargs)
        algorithm.setup(self.problem)
        return algorithm

    def test_ask_and_tell(self):
        pass


class TestBoTorchAlgorithm(TestAlgorithm):
    algorithm_class = DummyBoTorchAlgorithm
    algorithm_kwargs = {"n_init": 5}
    problem = DummyProblem(backend="torch", maximize=True)

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


class TestPymooAlgorithm(TestAlgorithm):
    algorithm_class = DummyPymooAlgorithm
    algorithm_kwargs = {"pop_size": 5}
    problem = DummyProblem(backend="pymoo", maximize=True)

    def test_ask_and_tell(self):
        algorithm = self._get_algorithm()
        # First iteration should initialize the population
        pop_x = algorithm.ask()
        self.assertIsInstance(pop_x, np.ndarray)
        self.assertEqual(pop_x.shape[0], algorithm.pop_size)
        self.assertEqual(pop_x.shape[1], len(self.problem.bounds))
        pop_y = self.problem(pop_x)
        self.assertIsInstance(pop_y, np.ndarray)
        self.assertEqual(pop_y.shape[0], algorithm.pop_size)
        algorithm.tell(pop_y)
        # Second iteration should create a new population
        new_pop_x = algorithm.ask()
        self.assertIsInstance(new_pop_x, np.ndarray)
        self.assertEqual(new_pop_x.shape[0], algorithm.pop_size)
        self.assertEqual(new_pop_x.shape[1], len(self.problem.bounds))
        new_pop_y = self.problem(new_pop_x)
        self.assertIsInstance(new_pop_y, np.ndarray)
        self.assertEqual(new_pop_y.shape[0], algorithm.pop_size)
        algorithm.tell(new_pop_y)
        # The number of individuals in the population should always be pop_size
        self.assertEqual(algorithm.train_x.shape[0], algorithm.pop_size)
        self.assertEqual(algorithm.train_y.shape[0], algorithm.pop_size)
