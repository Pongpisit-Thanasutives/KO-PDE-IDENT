import random
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination


def Ridge(X, y, lam=0):
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape
    lam = abs(lam)

    if lam != 0:
        # Solve (X^T X + lambda * I) beta = X^T y
        A = X.T @ X + lam * np.eye(d)
        b = X.T @ y
        coeff = np.linalg.solve(A, b)
    else:
        # Use least squares solver
        coeff, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Compute error manually (squared error)
    error = np.sum((y - X @ coeff) ** 2)

    return coeff, error


class PdeDiscoveryProblem(ElementwiseProblem):
    def __init__(
        self,
        n_poly,
        n_derivatives,
        n_modules,
        base_features,
        u_t,
        order_complexity=False,
        ridge_lambda=0,
        **kwargs
    ):
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, **kwargs)
        self.n_poly = n_poly
        self.n_derivatives = n_derivatives
        self.n_modules = n_modules
        self.base_features = base_features
        self.u_t = u_t
        self.sample_size = np.prod(self.u_t.shape)
        self.order_complexity = order_complexity
        self.ridge_lambda = ridge_lambda

    def _evaluate(self, X, out, *args, **kwargs):
        genome = X[0]
        coeff, mse = self.compute_genome_coefficient(genome)
        mse = mse / self.sample_size
        complexity_penalty = len(genome)
        if self.order_complexity:
            complexity_penalty += sum(sum(_) for _ in genome)
        out["F"] = [mse, complexity_penalty]

    def numericalize_genome(self, genome):
        return np.stack(
            [self.base_features[tuple(module)] for module in genome], axis=-1
        )

    def compute_genome_coefficient(self, genome):
        features = self.numericalize_genome(genome)
        features = features.reshape(-1, features.shape[-1])
        coeff, error = Ridge(features, self.u_t, self.ridge_lambda)
        return coeff, error

    def generate_module(self):
        return (random.randint(0, self.n_poly), random.randint(0, self.n_derivatives))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_order_complexity(self, order_complexity):
        self.order_complexity = order_complexity


class PopulationSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)
        X_set = set()
        i = 0
        while i < n_samples:
            n_modules = random.randint(1, problem.n_modules)
            genome = frozenset(problem.generate_module() for _ in range(n_modules))
            if len(genome) > 0 and genome not in X_set:
                X_set.add(genome)
                X[i, 0] = genome
                i += 1
        return X


class DuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, g1, g2):
        return g1.X[0] == g2.X[0]


class GenomeCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            Y[0, k, 0], Y[1, k, 0] = self.crossover_permutation(X[0, k, 0], X[1, k, 0])

        return Y

    def crossover_permutation(self, genome1, genome2):
        collection = list(genome1) + list(genome2)
        random.shuffle(collection)
        return (
            frozenset(collection[: len(genome1)]),
            frozenset(collection[len(genome1) :]),
        )


class GenomeMutation(Mutation):
    def __init__(self, add_rate=0.4, del_rate=0.5, order_rate=0.4):
        super().__init__()
        self.add_rate = add_rate
        self.del_rate = del_rate
        self.order_rate = order_rate

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if random.random() < self.add_rate:
                X[i, 0] = self.add_mutate(problem, X[i, 0])
            if random.random() < self.del_rate:
                X[i, 0] = self.del_mutate(problem, X[i, 0])
            if random.random() < self.order_rate:
                X[i, 0] = self.module_mutate(problem, X[i, 0])
        return X

    def add_mutate(self, problem, genome, max_iter=3):
        for _ in range(max_iter):
            new_module = problem.generate_module()
            if new_module not in genome:
                return genome.union(frozenset({new_module}))
        return genome

    def del_mutate(self, problem, genome, max_iter=3):
        genome = list(genome)
        lg = len(genome)
        if lg > 0:
            if lg == 1:
                for _ in range(max_iter):
                    new_module = problem.generate_module()
                    if new_module != genome[0]:
                        return frozenset({new_module})
            else:
                genome.pop(random.randint(0, lg - 1))
        return frozenset(genome)

    def module_mutate(self, problem, genome):
        if len(genome) == 0:
            return genome
        genome = set(genome)
        genome.remove(random.choice(list(genome)))
        for _ in range(3):
            new_module = problem.generate_module()
            if new_module not in genome:
                genome.add(new_module)
                return frozenset(genome)
        return frozenset(genome)
