from unittest import TestCase

import numpy as np
import torch
from pymoo.core.population import Population

from bbopy.sampling import FloatRandomSampling


class FloatRandomSamplingTest(TestCase):

    @staticmethod
    def _get_bounds():
        return [(-1, 1), (15, 150), (-10, -5)]

    def test_generate_numpy(self):
        bounds = self._get_bounds()
        sampler = FloatRandomSampling(backend="numpy", seed=42)
        x1 = sampler(bounds, 10)
        # Check the type
        self.assertIsInstance(x1, np.ndarray)
        # Check the shape
        self.assertEqual(x1.shape, (10, len(bounds)))
        # Check if it is contained in bounds
        for i, xbound in enumerate(bounds):
            self.assertTrue(all(x1[:, i] >= xbound[0]) and all(x1[:, i] <= xbound[1]))
        # Check if the seed works
        sampler = FloatRandomSampling(backend="numpy", seed=42)
        x2 = sampler(bounds, 10)
        self.assertListEqual(x1.tolist(), x2.tolist())

    def test_generate_pymoo(self):
        bounds = self._get_bounds()
        sampler = FloatRandomSampling(backend="pymoo", seed=42)
        x1 = sampler(bounds, 10)
        # Check the type
        self.assertIsInstance(x1, Population)
        x1 = x1.get("x")
        # Check the shape
        self.assertEqual(x1.shape, (10, len(bounds)))
        # Check if it is contained in bounds
        for i, xbound in enumerate(bounds):
            self.assertTrue(all(x1[:, i] >= xbound[0]) and all(x1[:, i] <= xbound[1]))
        # Check if the seed works
        sampler = FloatRandomSampling(backend="pymoo", seed=42)
        x2 = sampler(bounds, 10).get("x")
        self.assertListEqual(x1.tolist(), x2.tolist())

    def test_generate_torch(self):
        bounds = self._get_bounds()
        for dtype in [torch.int, torch.float, torch.double]:
            sampler = FloatRandomSampling(backend="torch", seed=42, dtype=dtype)
            x1 = sampler(bounds, 10)
            # Check the type
            self.assertIsInstance(x1, torch.Tensor)
            # Check the shape
            self.assertEqual(x1.shape, (10, len(bounds)))
            # Check if it is contained in bounds
            for i, xbound in enumerate(bounds):
                self.assertTrue(all(x1[:, i] >= xbound[0]) and all(x1[:, i] <= xbound[1]))
            # Check if the seed works
            sampler = FloatRandomSampling(backend="torch", seed=42, dtype=dtype)
            x2 = sampler(bounds, 10)
            self.assertListEqual(x1.tolist(), x2.tolist())

    def test_coherence_between_backends(self):
        bounds = self._get_bounds()
        sampler = FloatRandomSampling(backend="numpy", seed=42)
        xnumpy = sampler(bounds, 10)
        sampler = FloatRandomSampling(backend="pymoo", seed=42)
        xpymoo = sampler(bounds, 10).get("x")
        sampler = FloatRandomSampling(backend="torch", seed=42, dtype=torch.double)
        xtorch = sampler(bounds, 10)
        self.assertListEqual(xnumpy.tolist(), xpymoo.tolist())
        self.assertListEqual(xnumpy.tolist(), xtorch.tolist())
        self.assertListEqual(xpymoo.tolist(), xtorch.tolist())
