import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import eig
from scipy.integrate import solve_ivp


def plot_phase_portrait(alpha, matrix):
    """
    This function plots phase portrait in a streamplot, defined with the value of alpha and matrix A.
    
    :param w, X, Y: Set the range and resolution of the grid.
    :param A, alpha: parametrized 2x2 matrix based on the value of alpha.
    :param: title: get the titles, i.e. the type of phase portrait.
    :param eigenvalues: Calculate the eigenvalues of the matrix A
    :param U, V: Compute the vector field based on the matrix.
    :param gs: Define a grid specification for the subplot.
    
    """
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
        
    A = matrix["matrix"]
    title = matrix["title"]
    eigenvalues = eig(A)[0]
    
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(4, 4))   
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    
    # Plot the streamlines with varying density
    ax.streamplot(X, Y, U, V, density=[0.5, 1])
    # Set a title for the plot
    ax.set_title(f"alpha = {alpha}, λ1 = {eigenvalues[0]:.2f}, λ2 = {eigenvalues[1]:.2f}"); 
    
    # Set a legend for the plot and set the color
    ax.legend({title}, loc ="upper left")
    leg = ax.get_legend()
    for text in leg.get_texts():
        text.set_color("red")
        
    # Set the aspect ratio of the subplot to 1   
    ax.set_aspect(1)  
    # Save the plot
    plt.savefig('outputs/{}.png'.format(title))
    
    
def plot_phase_diagrams(alpha):
    """
    This function plots phase diagrams based on the value of alpha and the two given equations.
    
    :param w, x1, x2: Set the range and resolution of the grid.
    :param U, V: Define U and V based on the two equations, as well as values of alpha, x1, and x2.
    
    """
    w = 3
    x2, x1 = np.mgrid[-w:w:100j, -w:w:100j]    
    U = alpha * x1 - x2 - x1*(np.square(x1) + np.square(x2))
    V = x1 + alpha * x2 - x2*(np.square(x1) + np.square(x2))

    plt.figure(figsize=(4, 4)) 
    # Plot the streamlines with x1, x2, U, V, and the varying density
    plt.streamplot(x1, x2, U, V, density=1.5, color="cornflowerblue")
    # Set a title for the plot
    plt.title(f"alpha = {alpha}")
    # Save the plot
    plt.savefig('outputs/phase_diagram_alpha{}.png'.format(alpha))
    
 
def odefun(t, y):
    """
    This function is an implementation of a system of ordinary differential equations (ODEs),
    based on the two given equations.
    
    :param t: Independent variable (time).
    :param y: Array of dependent variables.
    :param m1, m2: Unpack the variables from the input vector y.
    :param f1, f2: Calculate the derivatives of f1 and f2 using the given formulas.
    
    Returns:
        The calculated derivatives f1 and f2.
    
    """
    alpha = 1
    m1, m2 = y
    f1 = alpha * m1 - m2 - m1*(np.square(m1) + np.square(m2))
    f2 = m1 + alpha * m2 - m2*(np.square(m1) + np.square(m2))
    return f1, f2
    
    
def compute_visualize_orbits(odefun, y0, color):
    """
    This function computes and visualizes the orbits with a given ODE function and a starting point.
    
    :param w, x1, x2: Set the range and resolution of the grid.
    :param U, V: Define U and V based on the two equations, as well as values of alpha, x1, and x2.
    :param odefun: ODE function to solve.
    :param y0: Initial conditions for the dependent variables.
    :param t_eval: Set the time span.
    
    """
    alpha = 1
    w = 3
    x2, x1 = np.mgrid[-w:w:100j, -w:w:100j]   
    U = alpha * x1 - x2 - x1*(np.square(x1) + np.square(x2))
    V = x1 + alpha * x2 - x2*(np.square(x1) + np.square(x2)) 
    
    # Solve the ODE using the given initial conditions and time span
    sol = solve_ivp(odefun, (0, 10), y0, t_eval=np.linspace(0, 10, 100))
    x, y = sol.y
    
    plt.figure(figsize=(4, 4))
    # Plot the streamlines with x1, x2, U, V, and the varying density
    plt.streamplot(x1, x2, U, V, color="lightgray")
    # Plot the computed trajectory
    plt.plot(x, y, color)
    # Plot the initial point with a red star marker
    plt.plot(y0[0], y0[1], marker='*', ms=15, color="red", alpha=0.5)
    # Set a title for the plot
    plt.title('orbit for start point{}'.format(y0))
    # Save the plot
    plt.savefig('outputs/orbit_startpoint{}.png'.format(y0))   
     
 
def compute_bifurcation_params():
    """
    This function computes and returns the values of alpha1_grid, alpha2_grid, x_grid, 
    which will be used in the function plot_bifurcation_surface().
    
    :param x_samples, alpha2_samples: Define the sample ranges for x and alpha2.
    :param x_grid, alpha2_grid: Create a grid of (x, alpha2) points.
    :param alpha1_grid: Create a grid of alpha1, whose values are computed using x and alpha2.
    
    Returns:
        The values of alpha1_grid, alpha2_grid, x_grid.
    
    """ 
    x_samples = np.linspace(-2, 2, 50)
    alpha2_samples = np.linspace(-2, 2, 50)    
    x_grid, alpha2_grid = np.meshgrid(x_samples, alpha2_samples)
    
    alpha1_grid = np.zeros_like(x_grid)   
    # Iterate over the indices of x_samples and alpha2_samples
    for i in range(len(x_samples)):
        for j in range(len(alpha2_samples)):
            # Retrieve the corresponding x and alpha2 values
            x = x_samples[i]
            alpha2 = alpha2_samples[j]
            # Solve for alpha1 using ẋ = 0
            alpha1_grid[j, i] = -alpha2 * x + x**3
                       
    return alpha1_grid, alpha2_grid, x_grid

 
def plot_bifurcation_surface(alpha1_grid, alpha2_grid, x_grid, elev, azim):
    """
    This function plots a 3D surface representing a cusp bifurcation with the given values of 
    alpha1_grid, alpha2_grid, and x_grid.

    """
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the bifurcation surface using the given values
    ax.plot_surface(alpha1_grid, alpha2_grid, x_grid, cmap='coolwarm', alpha=0.8)
    
    # Set labels for the x, y, and z axes
    ax.set_xlabel('α1')
    ax.set_ylabel('α2')
    ax.set_zlabel('x')
    # Change angle of 3D plot
    ax.view_init(elev, azim)
    # Save the plot
    plt.savefig('outputs/Cusp Bifurcation Surface_angle{}.png'.format((elev, azim)))
    plt.show()
    