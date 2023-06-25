import scipy
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp


def rbf(x, x_L, eps):
    
    """ compute the rbf for multiple x

    Args:
        x : the input value of rbf function
        x_L : the center of rbf function
        eps : the bandiwdth of rbf

    Returns:
        phi: the special functions φ: so-called radial basis functions
    """
    return np.exp(-cdist(x, x_L) ** 2 / eps ** 2)


def nonlinear_approximation_function(x, fx, L, eps):
    """
    1.Randomly selects L centers for the rbf.
    2.computes the transformation matrix C.
    3.use least-sqaures minimization to minimises |fx-phi(x)*C.T| and find matrix C

    Args:
        x : the coordinate x of data
        fx : the coordinate y of data
        eps : the denominator of phi
        L : the numbers of rbf

    Returns:
        C: the matric C which is calculate by the least-suqres minimization
        phi: the special functions φ: so-called radial basis functions
    """
    # randomly choose L data pointsx for the centers of the radial basis functions
    random.seed(6)
    center_indices = np.random.choice(range(x.shape[0]), replace=False, size=L)
    centers = x[center_indices]
    # transformation of the data matrix x to the new data matrix phi
    phi = rbf(x, centers, eps)
    C = np.linalg.lstsq(a=phi, b=fx, rcond=1e-5)[0]
    return C, phi

def lorenz_attractor(xyz0 = [10,10,10], T_end = 1000, sigma=10, beta=8/3, rho=28):
    """
    The function to get the trajectory of a lorenz attractor,all necessary parameters for lorenz attractor are given in the input 
    
    Args:
        xyz0 :  the coordinate of the initial point in x, y, z axis
        T_end :  the end of the simulation time, not the iteration count
        sigma :  parameter in lorenz attractor
        beta : parameter in lorenz attractor
        rho : parameter in lorenz attractor

    Returns:
       
        output : the output of solve_ivp based on the input lorenz system,which includes the list of time series, 
              list of coordinates in x, y, z axis along the whole simulation time
    """
    def lorenz_system( t, xyz0=xyz0 , sigma=sigma, beta=beta, rho=rho):
        """construct lorenz_system based on the mathematical formular"""
        x, y, z = xyz0
        return [sigma*(y-x),  x*(rho-z)-y,  x*y-beta*z]

    t_span = [0,T_end]
    t_eval = np.linspace(0,T_end,100000)
    output = solve_ivp(fun=lorenz_system, t_span=t_span, y0=xyz0, t_eval=t_eval)

    return output

    
def plot_3d_traj(output):
    """
    Function to plot the trajectory of the lorenz system

    Args:
        output : the result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(output.y[0], output.y[1], output.y[2], linewidth = 0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title('Lorenz Attractor')
    plt.savefig("figures/lorenz")
    plt.show()

    
def plot_bonus_traj(approx_x):
    """
    Function to plot the trajectory of the lorenz system in bonus task

    Args:
        output : the result obtained from the function lorenz attractor(), which contains the coordinates of points in 3 axis
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(approx_x[:, 0], approx_x[:, 1], approx_x[:, 2], s=1)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("x_3")
    plt.savefig("figures/bouns")
    plt.show()
    
def get_phi(x0_data, id_xl, current_x_data, eps):
    """
    function returns phi

    Args:
        x0_data: original data
        id_xl: index of random selected data
        current_x_data: current data (
        eps: epsilon ϵ
        
    Returns:
        matrix contains Phi
    """
    phi = rbf(current_x_data, x0_data[id_xl], eps)
    return phi
