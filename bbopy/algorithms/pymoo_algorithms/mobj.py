from pymoo.algorithms.moo.nsga2 import binary_tournament, NSGA2 as PymooNSGA2
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random, NSGA3 as PymooNSGA3
from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival
from pymoo.algorithms.moo.sms import SMSEMOA as PymooSMSEMOA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.reference_direction import ReferenceDirectionFactory

from bbopy.algorithms import PymooAlgorithm
from bbopy.sampling import Sampling, FloatRandomSampling


class NSGA2(PymooAlgorithm):
    name = "NSGA-II"

    def __init__(
            self,
            pop_size: int,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            survival=RankAndCrowding(),
            **kwargs
    ):
        super().__init__(pop_size)
        self._algo = PymooNSGA2(pop_size=pop_size,
                                sampling=sampling,
                                selection=selection,
                                crossover=crossover,
                                mutation=mutation,
                                survival=survival,
                                **kwargs)


class NSGA3(PymooAlgorithm):
    name = "NSGA-III"

    def __init__(
            self,
            ref_dirs: ReferenceDirectionFactory,
            pop_size: int = None,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover=SBX(eta=30, prob=1.0),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
            n_offsprings=None,
            **kwargs
    ):
        super().__init__(len(ref_dirs))
        self._algo = PymooNSGA3(ref_dirs=ref_dirs,
                                pop_size=pop_size,
                                sampling=sampling,
                                selection=selection,
                                crossover=crossover,
                                mutation=mutation,
                                eliminate_duplicates=eliminate_duplicates,
                                n_offsprings=n_offsprings,
                                **kwargs)


class SMSEMOA(PymooAlgorithm):
    name = "SMS-EMOA"

    def __init__(
            self,
            pop_size: int,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover=SBX(),
            mutation=PM(),
            survival=LeastHypervolumeContributionSurvival(),
            eliminate_duplicates=True,
            n_offsprings=None,
            normalize=True,
            **kwargs
    ):
        super().__init__(pop_size)
        self._algo = PymooSMSEMOA(pop_size=pop_size,
                                  sampling=sampling,
                                  selection=selection,
                                  crossover=crossover,
                                  mutation=mutation,
                                  survival=survival,
                                  eliminate_duplicates=eliminate_duplicates,
                                  n_offsprings=n_offsprings,
                                  normalize=normalize,
                                  **kwargs)
