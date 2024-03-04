import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf

from bbopy.algorithms.botorch_algorithms import MES
from test.algorithms.test_algorithm import TestBoTorchAlgorithm


class TestMES(TestBoTorchAlgorithm):
    algorithm_class = MES
    algorithm_kwargs = {"n_init": 5}

    def test_default_init(self):
        algorithm = self._get_algorithm()
        self.assertEqual(algorithm._acqf, qMaxValueEntropy)
        self.assertIn("candidate_set", algorithm._acqf_kwargs.keys())
        self.assertIsInstance(algorithm._acqf_kwargs["candidate_set"], torch.Tensor)
        self.assertEqual(algorithm._acqf_optimizer, optimize_acqf)
        self.assertDictEqual(algorithm._acqf_optimizer_kwargs, {"q": 1, "num_restarts": 10, "raw_samples": 50})
        self.assertEqual(algorithm._surrogate, SingleTaskGP)
        self.assertIn("input_transform", algorithm._surrogate_kwargs.keys())
        self.assertIn("outcome_transform", algorithm._surrogate_kwargs.keys())
        self.assertIsInstance(algorithm._surrogate_kwargs["input_transform"], Normalize)
        self.assertIsInstance(algorithm._surrogate_kwargs["outcome_transform"], Standardize)
