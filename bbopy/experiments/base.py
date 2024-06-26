import time

import pandas as pd

from bbopy.algorithms import Algorithm
from bbopy.problems import Problem
from bbopy.utils.output import Output
from bbopy.utils.verbose import Table
from bbopy.utils.verbose.display import Display


class Experiment:
    r"""An optimization experiment.

    It runs an optimization algorithm on a given problem and return the results.

    Attributes:
        problem (Problem): The Problem to be optimized.
        algorithm (Algorithm): The Algorithm used to optimize.
        termination (int): The termination criteria.
        display (Table): A Table used to organize the printed information during the run.
        output (Output): An Output object used to track the optimization results.

    Examples:
        Optimize the Ackley 3-dim function with a Genetic Algorithm (based on Pymoo):

        >>> from bbopy.algorithms.pymoo_algorithms import GeneticAlgorithm
        >>> from bbopy.experiments import Experiment
        >>> from bbopy.problems.sobj import Ackley
        >>>
        >>> prob = Ackley(dim=3, backend="pymoo")
        >>> algo = GeneticAlgorithm(pop_size=10)
        >>> exp_ga = Experiment(prob, algo, termination=30)
        >>> res_ga = exp_ga.optimize(verbose=True)

        Optimize the Ackley 3-dim function with Bayesian Optimization (based on BoTorch):

        >>> from botorch.acquisition import UpperConfidenceBound
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.models.transforms import Normalize
        >>> from bbopy.algorithms.botorch_algorithms import BO
        >>>
        >>> prob = Ackley(dim=3, backend="torch")
        >>> algo = BO(n_init=10,
        ...                             surrogate=SingleTaskGP,
        ...                             surrogate_kwargs={"input_transform": Normalize(prob.dim)},
        ...                             acquisition=UpperConfidenceBound,
        ...                             acquisition_kwargs={"beta": 3, "maximize": False})
        >>> exp_bo = Experiment(prob, algo, termination=10)
        >>> res_bo = exp_bo.optimize(verbose=True)
    """

    def __init__(self, problem: Problem, algorithm: Algorithm, termination: int) -> None:
        r"""Initializes problem and algorithm of the experiment.

        Args:
            problem: The Problem to be optimized.
            algorithm: The Algorithm used to optimize.
            termination:  The termination criteria.
        """
        self.display = None
        self.problem = problem
        self.algorithm = algorithm
        self.termination = termination

        self.output = Output()
        self.algorithm.setup(problem)
        self.display = Display(title=self.algorithm.name)

    def optimize(self, verbose: bool = False) -> pd.DataFrame:
        r"""Run the optimization loop to solve the problem with the selected algorithm.

        It uses the ask and tell interface of the algorithm and evaluate the new points at each iteration.
        The process is repeated until a termination criteria is met.

        Args:
            verbose: A boolean indicating the verbosity level.

        Returns:
            A pandas Dataframe with the optimization results, orginized according to the Output object.
        """
        if verbose:
            self.display.print_header()

        for i in range(self.termination + 1):
            start_time = time.perf_counter()
            x = self.algorithm.ask()
            ask_exec_time = time.perf_counter() - start_time
            y = self.problem(x)
            eval_exec_time = time.perf_counter() - (start_time + ask_exec_time)
            self.algorithm.tell(y)
            tell_exec_time = time.perf_counter() - (start_time + ask_exec_time + eval_exec_time)
            if verbose:
                self._update_display(i, sum([ask_exec_time, eval_exec_time, tell_exec_time]))
                self.display.println()
            self.output.append(index=i, x=x.tolist(), y=y.tolist(),
                               ask_times=ask_exec_time, eval_times=eval_exec_time, tell_times=tell_exec_time)
        return self.output.to_pandas()

    def _update_display(self, index: int, exec_time: float) -> None:
        r"""Update the information to be logged.

        Called at each iteration or generation of the optimization process.
        Add the new results to the Table object to be logged.

        Args:
            index: an integer representing the current iteration or generation.
            exec_time: the iteration or generation execution time.
        """
        n_eval = self.algorithm.n_eval
        best_seen = self.algorithm.best_seen(self.problem.maximize)
        self.display.update([index, n_eval, best_seen, exec_time])
