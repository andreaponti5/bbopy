from unittest import TestCase

import numpy as np
import torch
from pymoo.core.population import Population

from bbopy.sampling import NoSampling


class NoSamplingTest(TestCase):

    @staticmethod
    def _get_sample():
        return [[1] * 3, [2] * 3, [3] * 3]

    def test_generate_numpy(self):
        val = self._get_sample()
        sampler = NoSampling(val=val, backend="numpy")
        gen_val = sampler()
        self.assertIsInstance(gen_val, np.ndarray)
        self.assertListEqual(gen_val.tolist(), val)

    def test_generate_pymoo(self):
        val = self._get_sample()
        sampler = NoSampling(val=val, backend="pymoo")
        gen_val = sampler()
        self.assertIsInstance(gen_val, Population)
        self.assertListEqual(gen_val.get("x").tolist(), val)

    def test_generate_torch(self):
        for dtype in [torch.int, torch.float, torch.double]:
            val = self._get_sample()
            sampler = NoSampling(val=val, backend="torch", dtype=dtype)
            gen_val = sampler()
            self.assertIsInstance(gen_val, torch.Tensor)
            self.assertEqual(gen_val.dtype, dtype)
            self.assertListEqual(gen_val.tolist(), val)
