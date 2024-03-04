from abc import ABC, abstractmethod
from unittest import TestCase

import pandas as pd

from bbopy.algorithms import Algorithm
from bbopy.experiments import Experiment
from bbopy.problems import Problem
from test.algorithms.test_algorithm import DummyBoTorchAlgorithm, DummyPymooAlgorithm
from test.problems.test_problem import DummyProblem


class TestExperiment(TestCase):
    algorithm: Algorithm
    problem: Problem


class TestExperimentBoTorch(TestExperiment):
    algorithm = DummyBoTorchAlgorithm(n_init=5)
    problem = DummyProblem(backend="torch")

    def test_optimize(self):
        _test_optimize(self)


class TestExperimentPymoo(TestExperiment):
    algorithm = DummyPymooAlgorithm(pop_size=5)
    problem = DummyProblem(backend="pymoo")

    def test_optimize(self):
        _test_optimize(self)


def _test_optimize(test_case: TestExperiment):
    termination = 10
    exp = Experiment(problem=test_case.problem, algorithm=test_case.algorithm, termination=termination)
    res = exp.optimize(verbose=False)
    test_case.assertIsInstance(res, pd.DataFrame)
    test_case.assertEqual(res.shape[0], test_case.algorithm.n_eval)
    test_case.assertEqual(res.shape[1], 1 + test_case.problem.dim + 1 + 3)
