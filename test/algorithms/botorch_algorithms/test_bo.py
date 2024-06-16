from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

from bbopy.algorithms.botorch_algorithms import BO
from test.algorithms.test_algorithm import TestBoTorchAlgorithm


class TestBayesianOptimization(TestBoTorchAlgorithm):
    algorithm_class = BO
    algorithm_kwargs = {"n_init": 5}

    def test_default_init(self):
        algorithm = self._get_algorithm()
        self.assertEqual(algorithm._acqf, UpperConfidenceBound)
        self.assertDictEqual(algorithm._acqf_kwargs, {"beta": 3})
        self.assertEqual(algorithm._acqf_optimizer, optimize_acqf)
        self.assertDictEqual(algorithm._acqf_optimizer_kwargs, {"q": 1, "num_restarts": 10, "raw_samples": 50})
        self.assertEqual(algorithm._surrogate, SingleTaskGP)
        self.assertDictEqual(algorithm._surrogate_kwargs, {})
