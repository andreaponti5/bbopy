from pymoo.algorithms.moo.moead import ParallelMOEAD as PymooMOEAD
from pymoo.algorithms.moo.nsga2 import binary_tournament, NSGA2 as PymooNSGA2
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random, NSGA3 as PymooNSGA3
from pymoo.algorithms.moo.sms import LeastHypervolumeContributionSurvival
from pymoo.algorithms.moo.sms import SMSEMOA as PymooSMSEMOA
from pymoo.core.crossover import Crossover
from pymoo.core.decomposition import Decomposition
from pymoo.core.mutation import Mutation
from pymoo.core.selection import Selection
from pymoo.core.survival import Survival
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
            selection: Selection = TournamentSelection(func_comp=binary_tournament),
            crossover: Crossover = SBX(eta=15, prob=0.9),
            mutation: Mutation = PM(eta=20),
            survival: Survival = RankAndCrowding(),
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
            selection: Selection = TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover: Crossover = SBX(eta=30, prob=1.0),
            mutation: Mutation = PM(eta=20),
            eliminate_duplicates: bool = True,
            n_offsprings: int = None,
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


class MOEAD(PymooAlgorithm):
    name = "MOEAD"

    def __init__(
            self,
            ref_dirs: ReferenceDirectionFactory,
            n_neighbors: int = 20,
            decomposition: Decomposition = None,
            prob_neighbor_mating: float = 0.9,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            crossover=SBX(prob=1.0, eta=20),
            mutation=PM(prob_var=None, eta=20),
            **kwargs

    ):
        super().__init__(len(ref_dirs))
        self._algo = PymooMOEAD(ref_dirs=ref_dirs,
                                n_neighbors=n_neighbors,
                                decomposition=decomposition,
                                prob_neighbor_mating=prob_neighbor_mating,
                                sampling=sampling,
                                crossover=crossover,
                                mutation=mutation,
                                **kwargs)


class SMSEMOA(PymooAlgorithm):
    name = "SMS-EMOA"

    def __init__(
            self,
            pop_size: int,
            sampling: Sampling = FloatRandomSampling(backend="pymoo"),
            selection: Selection = TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover: Crossover = SBX(),
            mutation: Mutation = PM(),
            survival: Survival = LeastHypervolumeContributionSurvival(),
            eliminate_duplicates: bool = True,
            n_offsprings: int = None,
            normalize: bool = True,
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
