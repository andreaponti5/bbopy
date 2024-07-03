import copy
import math
from abc import ABC
from typing import Optional, Union

import numpy as np
import torch

from bbopy.problems.base import MultiObjectiveProblem


class DTLZ(MultiObjectiveProblem, ABC):
    n_constr: int = 0
    _ref_val: float

    def __init__(
            self,
            backend: str,
            dim: int,
            n_obj: int,
            maximize: Optional[bool] = False,
    ) -> None:
        if dim <= n_obj:
            raise ValueError(f"dim must be > n_obj, but got {dim} and {n_obj}.")
        super().__init__(backend=backend, dim=dim, n_obj=n_obj, maximize=maximize)
        self.bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._ref_point = [self._ref_val for _ in range(self.n_obj)]
        self.k = self.dim - self.n_obj + 1


class DTLZ1(DTLZ):
    _ref_val: float = 400.0

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()
        x_m = x[..., -self.k:]
        x_m_minus_half = x_m - 0.5
        sum_term = (
                x_m_minus_half**2 - back.cos(20 * math.pi * x_m_minus_half)
        ).sum(**kwargs)
        g_x_m = 100 * (self.k + sum_term)
        g_x_m_term = 0.5 * (1 + g_x_m)
        fs = []
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = g_x_m_term * x[..., :idx].prod(**kwargs)
            if i > 0:
                f_i *= 1 - x[..., idx]
            fs.append(f_i)
        return back.stack(fs, **kwargs)


class DTLZ2(DTLZ):
    _ref_val: float = 1.1
    _alpha: int = 1

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()

        x_m = x[..., -self.k:]
        g_x = ((x_m - 0.5) ** 2).sum(**kwargs)
        g_x_plus1 = 1 + g_x
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = copy.deepcopy(g_x_plus1)
            f_i *= back.cos(x[..., :idx] ** self._alpha * pi_over_2).prod(**kwargs)
            if i > 0:
                f_i *= back.sin(x[..., idx] ** self._alpha * pi_over_2)
            fs.append(f_i)
        return back.stack(fs, **kwargs)


class DTLZ3(DTLZ):
    _ref_val: float = 10000.0

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()

        x_m = x[..., -self.k:]
        g_x = 100 * (
                x_m.shape[-1]
                + ((x_m - 0.5) ** 2 - back.cos(20 * math.pi * (x_m - 0.5))).sum(**kwargs)
        )
        g_x_plus1 = 1 + g_x
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.n_obj):
            idx = self.n_obj - 1 - i
            f_i = copy.deepcopy(g_x_plus1)
            f_i *= back.cos(x[..., :idx] * pi_over_2).prod(**kwargs)
            if i > 0:
                f_i *= back.sin(x[..., idx] * pi_over_2)
            fs.append(f_i)
        return back.stack(fs, **kwargs)


class DTLZ4(DTLZ2):
    _alpha: int = 100


class DTLZ5(DTLZ):
    _ref_val = 10.0

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()
        x_m = x[..., -self.k:]
        x_ = x[..., : -self.k]
        g_x = ((x_m - 0.5) ** 2).sum(**kwargs)
        theta = 1 / (2 * (1 + g_x.reshape(-1, 1))) * (1 + 2 * g_x.reshape(-1, 1) * x_)
        theta = back.concatenate([x[..., :1], theta[..., 1:]], **kwargs)
        fs = []
        pi_over_2 = math.pi / 2
        g_x_plus1 = g_x + 1
        for i in range(self.n_obj):
            f_i = copy.deepcopy(g_x_plus1)
            f_i *= back.cos(theta[..., : theta.shape[-1] - i] * pi_over_2).prod(**kwargs)
            if i > 0:
                f_i *= back.sin(theta[..., theta.shape[-1] - i] * pi_over_2)
            fs.append(f_i)
        return back.stack(fs, **kwargs)


class DTLZ7(DTLZ):
    _ref_val = 15.0

    def _evaluate(
            self,
            x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        back, kwargs = self._get_backend()
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[..., i])
        f = back.stack(f, **kwargs)

        g_x = 1 + 9 / self.k * back.sum(x[..., -self.k:], **kwargs)
        h = self.n_obj - back.sum(
            f / (1 + g_x.reshape(-1, 1)) * (1 + back.sin(3 * math.pi * f)), **kwargs
        )
        return back.concatenate([f, ((1 + g_x) * h).reshape(-1, 1)], **kwargs)
