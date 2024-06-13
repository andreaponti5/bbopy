# Define Problems

## Basics
Problems in BBOPy are defined as **callable classes**. 
The basic attributes of a problem are:

- `dim` (_int_): The number of variables. In some problems this is fixed, in others can be specified.
- `n_obj` (_int_): The number of objectives. In some problems this is fixed, in others can be specified.
- `n_constr` (_int_): The number of constraints. This is generally fixed.
- `bounds` (_List[Tuple[float, float]]_): The lower and upper bounds of each variable.
- `maximize` (_bool_): A boolean indicator that represents whether the problem has to be minimized or maximized.

All problems need to define the problem logic in the `_evaluate` method. It takes an iterable 
(numpy ndarray or torch tensor) with the points to evaluate: the shape is `n_point x dim`. The output is an iterable 
(numpy ndarray or torch tensor) with the objective(s) value for each input point: the shape is `n_point x n_obj`.

To make the problems adaptable to both, BoTorch and Pymoo, algorithms the backend used for the vector-operations can 
be changed accordingly.

- **Pymoo algorithms** use numpy arrays and to allow the usage of gradients, the backend used in autograd-numpy. 
When defining a problem that will be optimized with Pymoo, you need to pass `backend="pymoo"`.
In this way, the result of evaluating the problem will be a numpy array.
- **BoTorch algorithms** use torch tensors. When defining a problem that will be optimized with BoTorch, 
you need to pass `backend="torch"`. In this way, the result of evaluating the problem will be a torch tensor.

A bunch of test problems have already been implemented in the `bbopy.problems` module. 
Problems are divided into two submodules:

- `bbopy.problem.sobj`: contains single-objective test problems.
- `bbopy.problem.mobj`: contains multi-objectives test problems.

Some of these problems allow you to specify parameters like the number of variables (`dim`), 
the number of objectives (`n_obj`) or additional problem-specific parameters. 
The following example shows how to use the Ackley test function:

```py
import numpy as np
import torch

from bbopy.problems.sobj import Ackley

prob = Ackley(dim=5, backend="pymoo")
prob(np.zeros((1, 5)))
# > Output: array([[4.4408921e-16]])

prob = Ackley(dim=5, backend="torch")
prob(torch.zeros((1, 5), dtype=torch.float64))
# > Output: tensor([[4.4409e-16]], dtype=torch.float64)
```

## Define your own Problem
You can easily create your own problem, extending one of the base classes: `Problem`, `SingleObjectiveProblem` or 
`MultiObjectiveProblem`.

Suppose you want to create a problem with two objectives, the first is to minimize the _sine_ and the second is to minimize the _cosine_.
You need to create a class extending `MultiObjectiveProblem` as follows.

```py
from bbopy.problems.base import MultiObjectiveProblem


class SinCosProblem(MultiObjectiveProblem):
    dim = 1
    n_obj = 2
    
    def __init__(self, backend: str):
        super().__init__(backend)

    def _evaluate(self, x):
        back, kwargs = self._get_backend()
        f = back.zeros((x.shape[0], self.n_obj))
        f[:, 0] = back.sin(x)
        f[:, 1] = back.cos(x)
        return f
```
