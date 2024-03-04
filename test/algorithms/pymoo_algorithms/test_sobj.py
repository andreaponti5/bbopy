import numpy as np

from bbopy.algorithms.pymoo_algorithms import GeneticAlgorithm, DifferentialEvolution, EvolutionaryStrategy, CMAES
from test.algorithms.test_algorithm import TestPymooAlgorithm


class TestGeneticAlgorithm(TestPymooAlgorithm):
    algorithm_class = GeneticAlgorithm
    algorithm_kwargs = {"pop_size": 5}


class TestDifferentialEvolution(TestPymooAlgorithm):
    algorithm_class = DifferentialEvolution
    algorithm_kwargs = {"pop_size": 5}


class TestCMAES(TestPymooAlgorithm):
    algorithm_class = CMAES
    algorithm_kwargs = {}

    def test_ask_and_tell(self):
        algorithm = self._get_algorithm()
        # First iteration should initialize the population
        pop_x = algorithm.ask()
        self.assertIsInstance(pop_x, np.ndarray)
        self.assertEqual(pop_x.shape[0], algorithm.pop_size)
        self.assertEqual(pop_x.shape[1], len(self.problem.bounds))
        pop_y = self.problem(pop_x)
        self.assertIsInstance(pop_y, np.ndarray)
        self.assertEqual(pop_y.shape[0], algorithm.pop_size)
        algorithm.tell(pop_y)
        # Second iteration should create a new population
        n_offspring = 4 + int(3 * np.log(len(self.problem.bounds)))
        new_pop_x = algorithm.ask()
        self.assertIsInstance(new_pop_x, np.ndarray)
        self.assertEqual(new_pop_x.shape[0], n_offspring)
        self.assertEqual(new_pop_x.shape[1], len(self.problem.bounds))
        new_pop_y = self.problem(new_pop_x)
        self.assertIsInstance(new_pop_y, np.ndarray)
        self.assertEqual(new_pop_y.shape[0], n_offspring)
        algorithm.tell(new_pop_y)
        # The number of individuals in the new population is `4 + int(3 * np.log(dim))`
        self.assertEqual(algorithm.train_x.shape[0], n_offspring)
        self.assertEqual(algorithm.train_y.shape[0], n_offspring)


class TestEvolutionaryStrategy(TestPymooAlgorithm):
    algorithm_class = EvolutionaryStrategy
    algorithm_kwargs = {"pop_size": 5}

    def test_ask_and_tell(self):
        algorithm = self._get_algorithm()
        # First iteration should initialize the population
        pop_x = algorithm.ask()
        self.assertIsInstance(pop_x, np.ndarray)
        self.assertEqual(pop_x.shape[0], algorithm.pop_size)
        self.assertEqual(pop_x.shape[1], len(self.problem.bounds))
        pop_y = self.problem(pop_x)
        self.assertIsInstance(pop_y, np.ndarray)
        self.assertEqual(pop_y.shape[0], algorithm.pop_size)
        algorithm.tell(pop_y)
        # Second iteration should create a new population
        new_pop_x = algorithm.ask()
        self.assertIsInstance(new_pop_x, np.ndarray)
        self.assertEqual(new_pop_x.shape[0], algorithm.pop_size * 2)
        self.assertEqual(new_pop_x.shape[1], len(self.problem.bounds))
        new_pop_y = self.problem(new_pop_x)
        self.assertIsInstance(new_pop_y, np.ndarray)
        self.assertEqual(new_pop_y.shape[0], algorithm.pop_size * 2)
        algorithm.tell(new_pop_y)
        # The number of individuals in the new population is two `2 x pop_size`
        self.assertEqual(algorithm.train_x.shape[0], algorithm.pop_size * 2)
        self.assertEqual(algorithm.train_y.shape[0], algorithm.pop_size * 2)
