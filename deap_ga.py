import random
import numpy as np

def generate_module(n_poly, n_derivatives):
    return (random.randint(0, n_poly), random.randint(0, n_derivatives))

def generate_ode_module(n_poly, n_vars):
    module = [0 for _ in range(n_vars)]
    for i in range(n_vars):
        deg = random.randint(0, n_poly)
        module[i] = deg
        n_poly -= deg
        if n_poly <= 0:
            break
    return tuple(module)

# container = frozenset, func = generate_module, n = max complexity of a generated PDE
def myInitRepeat(container, func, n):
    return container(func() for _ in range(random.randint(1, n)))

def numericalize_module(module, base_features):
    assert type(module) is tuple
    return base_features[module]

def numericalize_genome(genome, base_features):
    return np.stack([numericalize_module(module, base_features)
                     for module in genome], axis=-1)

def compute_genome_coefficient(genome, base_features, target):
    features = numericalize_genome(genome, base_features)
    n_features = features.shape[-1]
    target = target.reshape(-1, 1)
    features = features.reshape(-1, n_features)
    coeff, error, _, _ = np.linalg.lstsq(features, target, rcond=None)
    return coeff, error[0]

def evaluate_genome(genome, base_features, target, epsilon=0):
    coeff, mse = compute_genome_coefficient(genome, base_features, target)
    mse = mse / np.prod(target.shape)
    return mse, epsilon*len(genome)

def crossover(genome1, genome2):
    genome1, genome2 = list(genome1), list(genome2)
    idx1 = random.randint(0, len(genome1)-1)
    idx2 = random.randint(0, len(genome2)-1)
    genome1[idx1], genome2[idx2] = genome2[idx2], genome1[idx1]
    return frozenset(genome1), frozenset(genome2)

def crossover_permutation(genome1, genome2):
    collection = list(genome1) + list(genome2)
    random.shuffle(collection)
    return frozenset(collection[:len(genome1)]), frozenset(collection[len(genome1):])

def add_mutate(genome, n_poly, n_derivatives, max_iter=3):
    for _ in range(max_iter):
        new_module = generate_module(n_poly, n_derivatives)
        if new_module not in genome:
            return genome.union(frozenset({new_module})),
    return genome,

def del_mutate(genome):
    genome = list(genome)
    lg = len(genome)
    if lg > 0:
        genome.pop(random.randint(0, lg-1))
    return frozenset(genome),

def module_mutate(genome, n_poly, n_derivatives):
    if len(genome) == 0:
        return genome,
    genome = set(genome)
    genome.remove(random.choice(list(genome)))
    for _ in range(3):
        new_module = generate_module(n_poly, n_derivatives)
        if new_module not in genome:
            genome.add(new_module)
            return frozenset(genome),
    return frozenset(genome),

def order_mutate(genome, n_poly, n_derivatives):
    genome = list(genome)
    lg = len(genome)
    if lg > 0:
        i = random.randint(0, lg-1)
        if len(genome[i]) > 0:
            genome[i] = list(genome[i])
            j = random.randint(0, len(genome[i])-1)
            if genome[i][j] == 0:
                if j == 0:
                    genome[i][j] = random.sample([i for i in range(0, genome[i][j])] + [i for i in range(genome[i][j]+1, n_poly+1)], 1)[0]
                else:
                    genome[i][j] = random.sample([i for i in range(0, genome[i][j])] + [i for i in range(genome[i][j]+1, n_derivatives+1)], 1)[0]
            else:
                genome[i][j] -= 1
            genome[i] = tuple(genome[i])
    return frozenset(genome),

