import warnings
from typing import Optional, Dict, Any, Callable, Type

import torch
from botorch.acquisition import qMaxValueEntropy, AcquisitionFunction, qLowerBoundMaxValueEntropy
from botorch.exceptions import BotorchTensorDimensionWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf

from bbopy.algorithms.botorch_algorithms import BO
from bbopy.sampling import Sampling, FloatRandomSampling


class MES(BO):
    r"""A Bayesian Optimization algorithm using the Max-value Entropy Search (MES) acquisition function."""
    name: str = "MES"

    def __init__(
            self,
            n_init: int,
            acquisition: Optional[Type[AcquisitionFunction]] = qMaxValueEntropy,
            acquisition_optimizer: Optional[Callable] = None,
            surrogate_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ):
        warnings.simplefilter("ignore", BotorchTensorDimensionWarning)
        super().__init__(
            n_init,
            surrogate=SingleTaskGP,
            acquisition=acquisition,
            acquisition_optimizer=acquisition_optimizer,
            surrogate_kwargs=surrogate_kwargs,
            acquisition_kwargs=acquisition_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            sampling=sampling
        )

    def _setup(self) -> None:
        r"""Sets the default surrogate model and acquisition function, if they are not specified."""
        candidate_set = torch.rand(
            1000, self._bounds.size(1), device=self._bounds.device, dtype=self._bounds.dtype
        )
        candidate_set = self._bounds[0] + (self._bounds[1] - self._bounds[0]) * candidate_set
        self._acqf_kwargs = self._acqf_kwargs or {"candidate_set": candidate_set}
        self._acqf_optimizer = self._acqf_optimizer or optimize_acqf
        self._acqf_optimizer_kwargs = self._acqf_optimizer_kwargs or {"q": 1, "num_restarts": 10, "raw_samples": 50}
        self._surrogate_kwargs = self._surrogate_kwargs or {"input_transform": Normalize(self._bounds.shape[-1],
                                                                                         bounds=self._bounds),
                                                            "outcome_transform": Standardize(1)}
        self._bounds = self._bounds.to(**self._sampling.tkwargs)


class GIBBON(MES):
    r"""A Bayesian Optimization algorithm using the GIBBON acquisition function."""
    name: str = "GIBBON"

    def __init__(
            self,
            n_init: int,
            acquisition: Optional[Type[AcquisitionFunction]] = qLowerBoundMaxValueEntropy,
            acquisition_optimizer: Optional[Callable] = None,
            surrogate_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ):
        warnings.simplefilter("ignore", BotorchTensorDimensionWarning)
        super().__init__(
            n_init,
            acquisition=acquisition,
            acquisition_optimizer=acquisition_optimizer,
            surrogate_kwargs=surrogate_kwargs,
            acquisition_kwargs=acquisition_kwargs,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            sampling=sampling
        )
