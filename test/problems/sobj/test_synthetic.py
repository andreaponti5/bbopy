from bbopy.problems.sobj import Ackley, Beale, Bukin, Branin
from test.problems.test_problem import TestSingleObjectiveProblem


class TestAckley(TestSingleObjectiveProblem):
    problem = Ackley
    problem_kwargs = {"dim": 5}


class TestBeale(TestSingleObjectiveProblem):
    problem = Beale


class TestBranin(TestSingleObjectiveProblem):
    problem = Branin


class TestBukin(TestSingleObjectiveProblem):
    problem = Bukin
