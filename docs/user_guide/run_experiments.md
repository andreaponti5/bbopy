# Run Experiments

## Using BoTorch algorithms
First, you need to define the problem to optimize. By default, all the problems have to be minimized.
If you want the problem defined as a maximization problem, you can specify `maximize=True`. 
Let's use the `Ackley` test problem defined in 5 dimensions. 
When using BoTorch algorithms you need to specify `backend="torch"`.
```py
from bbopy.problems.sobj import Ackley

prob = Ackley(dim=5, maximize=True, backend="torch")
```
Then, let's define a simple Bayesian Optimization algorithm. 
By default, a `SingleTaskGP` is used as surrogate model and `UpperConfidenceBound` is used as acquisition function.
You need to specify the number of initial points `n_init` to start up the optimization loop. 
These points are sampled using the strategy specified as parameter `sampling`, by default `FloatRandomSampling` is used.
```py
from bbopy.algorithms.botorch_algorithms import BayesianOptimization

algo = BayesianOptimization(n_init=5)
```
Finally, you can define the expirement using the previously defined problem and algorithm.
In addition, you need to specify the `termination` criteria. 
By default, with BoTorch algorithms, this is the number of iterations.
```py
from bbopy.experiments import Experiment

exp = Experiment(prob, algo, termination=20)
res = exp.optimize(verbose=True)
```

## Using Pymoo algorithms
As in the previous example, you need to define the problem to optimize. 
When using Pymoo algorithms you need to specify `backend="pymoo"`. Pymoo algorithms consider minimization problems.
```py
from bbopy.problems.sobj import Ackley

prob = Ackley(dim=5, maximize=False, backend="pymoo")
```
Then, let's define a simple Genetic Algorithm.
You can specify the evolutionary operators to use, by default a Polynomial Mutation and a Simulated Binary Crossover are used.
You can also set the population size `pop_size` and the number of offsprings to generate at each generation `n_offsprings`.
```py
from bbopy.algorithms.pymoo_algorithms import GeneticAlgorithm

algo = GeneticAlgorithm(pop_size=5)
```
Finally, you can define the expirement using the previously defined problem and algorithm.
In addition, you need to specify the `termination` criteria.
By default, with Pymoo algorithms, this is the number of generations.
```py
from bbopy.experiments import Experiment

exp = Experiment(prob, algo, termination=20)
res = exp.optimize(verbose=True)
```