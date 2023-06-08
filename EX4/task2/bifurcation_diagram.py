import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, core
from typing import Tuple
from sympy.solvers import solve


def bifurcation_diagram(funtion_solution, min_alpha, max_alpha, alpha_gap):
   
    """
    Plots the bifurcation diagram with a given function.
    
    Args:
        funtion_solution: the function of dynamic system.
        min_alpha: the lowest value of alpha.
        max_alpha: the highest value of alpha.
        alpha_gap: the gap value of alpha, which control the numbers of alpha.

    """
    
    x = Symbol('x')
    alphas = np.arange(min_alpha, max_alpha, alpha_gap)
    
    x_solutions = {}
    alphas_location = {}
    
    # calculate each aplha to get the whole states of dynamic systems
    for alpha in alphas:
        sol = solve(eval(funtion_solution), x)
        for i, single_sol in enumerate(sol):
            if i not in x_solutions:
                x_solutions[i] = [single_sol]
                alphas_location[i] = [alpha]
            else:
                x_solutions[i].append(single_sol)
                alphas_location[i].append(alpha)
                  
    # jut plot the resonable results, remove the complex number results.
    for i in sorted(x_solutions.keys()):
        for j in range(len(x_solutions[i])):
            if not isinstance(x_solutions[i][j], core.numbers.Float) and not isinstance(x_solutions[i][j],
                                                                                         core.numbers.Integer):
                x_solutions[i][j] = None
        plt.scatter(alphas_location[i], x_solutions[i])
        
        
    plt.xlim(alphas[0], alphas[-1])
    plt.xlabel('alpha')
    plt.ylabel('x')
    plt.title(funtion_solution)
    plt.show()