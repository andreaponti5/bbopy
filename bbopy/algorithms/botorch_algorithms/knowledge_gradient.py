from typing import Optional, Callable, Dict, Any, Tuple

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qKnowledgeGradient, PosteriorMean, qMultiFidelityKnowledgeGradient, \
    FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from gpytorch import ExactMarginalLogLikelihood

from bbopy.algorithms.botorch_algorithms.vanilla import BO
from bbopy.sampling import FloatRandomSampling, Sampling


class KG(BO):
    name = "KG"

    def __init__(
            self,
            n_init: int,
            acquisition_optimizer: Optional[Callable] = None,
            surrogate_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ):
        super().__init__(n_init=n_init,
                         surrogate=SingleTaskGP,
                         acquisition=qKnowledgeGradient,
                         acquisition_optimizer=acquisition_optimizer,
                         surrogate_kwargs=surrogate_kwargs,
                         acquisition_kwargs=acquisition_kwargs,
                         acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
                         sampling=sampling)

    def _setup(self) -> None:
        self._acqf_kwargs = self._acqf_kwargs or {}
        self._acqf_optimizer = self._acqf_optimizer or optimize_acqf
        self._acqf_optimizer_kwargs = self._acqf_optimizer_kwargs or {"q": 1, "num_restarts": 10, "raw_samples": 50}
        self._surrogate_kwargs = self._surrogate_kwargs or {}
        self._bounds = self._bounds.to(**self._sampling.tkwargs)

    def ask(self) -> torch.Tensor:
        if self._train_x is None:
            self._train_x = self._sampling(self.bounds, self.n_init)
            return self._train_x

        x_init = gen_one_shot_kg_initial_conditions(
            acq_function=self.acquisition,
            bounds=self._bounds,
            **self._acqf_optimizer_kwargs
        )
        candidates, acq_value = self._acqf_optimizer(acq_function=self.acquisition,
                                                     bounds=self._bounds,
                                                     batch_initial_conditions=x_init,
                                                     **self._acqf_optimizer_kwargs)
        new_x = candidates.detach()
        self._train_x = torch.cat([self._train_x, new_x])
        return new_x

    def tell(self, y) -> None:
        if self._train_y is None:
            self._train_y = y
        else:
            self._train_y = torch.cat([self._train_y, y])
        self.model = self._surrogate(self._train_x, self._train_y, **self._surrogate_kwargs)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        argmax_pmean, max_pmean = self._get_max_posterior_mean()
        self.acquisition = self._acqf(self.model, current_value=max_pmean, **self._acqf_kwargs)

    def _get_max_posterior_mean(self) -> Tuple:
        return self._acqf_optimizer(
            acq_function=PosteriorMean(self.model),
            bounds=self._bounds,
            **self._acqf_optimizer_kwargs
        )


class MFKG(KG):
    name = "MF-KG"

    def __init__(
            self,
            n_init: int,
            acquisition_optimizer: Optional[Callable] = None,
            surrogate_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_kwargs: Optional[Dict[str, Any]] = None,
            acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
            sampling: Sampling = FloatRandomSampling(backend="torch", dtype=torch.double)
    ):
        super().__init__(n_init=n_init,
                         acquisition_optimizer=acquisition_optimizer,
                         surrogate_kwargs=surrogate_kwargs,
                         acquisition_kwargs=acquisition_kwargs,
                         acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
                         sampling=sampling)
        self._surrogate = SingleTaskMultiFidelityGP
        self._acqf = qMultiFidelityKnowledgeGradient

    def _setup(self) -> None:
        super()._setup()
        if "data_fidelities" not in self._surrogate_kwargs:
            self._surrogate_kwargs["data_fidelities"] = [-1]
        self.data_fidelities = self._surrogate_kwargs["data_fidelities"]
        self.target_fidelities = {fid: 1.0 for fid in self.data_fidelities}
        self._acqf_kwargs["project"] = self._project

    def _get_max_posterior_mean(self) -> Tuple:
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(self.model),
            d=len(self.bounds),
            columns=self.data_fidelities,
            values=[1] * len(self.data_fidelities),
        )
        return optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self._bounds[:, :-1],
            **self._acqf_optimizer_kwargs
        )

    def _project(self, x):
        return project_to_target_fidelity(X=x, target_fidelities=self.target_fidelities)

    def best_seen(self, maximize: bool = False) -> float:
        max_fidelity = self.train_x[..., self.data_fidelities].max()
        fid_indeces = torch.where(self.train_x[..., self.data_fidelities] == max_fidelity)
        fid_objectives = self.train_y[fid_indeces]
        return fid_objectives.max() if maximize else fid_objectives.min()
