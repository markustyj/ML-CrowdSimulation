import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

def diffusion_maps(data_set, L):
    """
    :param data_set : the given data set
    :type data_set : np.array
    :param L : the number of eigenfunctions
    :type L : int
    :return eigenvalues and eigenvectors  
    """ 
    
    #print("dataset:\n", data_set)
    #print(data_set.shape)
    
    # 1. Compute the distance matrix using KDTree
    # Build the KDTree
    tree = KDTree(data_set) 
    dist_matrix = tree.sparse_distance_matrix(tree, max_distance=np.inf).toarray()
    #print("Distance matrix: \n" , dist_matrix)
    
    
    # 2. Compute the epsilon 
    epsilon = 0.05 * np.max(dist_matrix)
    #print("epsilon:" , epsilon)
    
    
    # 3. Compute the kernel matrix W
    W = np.exp(- dist_matrix ** 2 / epsilon)
    #print("W:" , W)

 
    # 4. Compute the inverse of diagonal normalization matrix P^(-1)
    # In this step the inverse of P is calculated directly so that the next step can be used without the denominator being zero
    P = np.diag(np.reciprocal(np.sum(W, axis=1)))
    #print("the inverse of P" , P)
    

    # 5. normalize to form the kernel matrix K = P^{-1} * W * P^{-1}
    K = np.dot(P, np.dot(W, P))
    #print("K" , K)
    

    # 6. calculate the inverse of the diagonal normalization matrix Q^(-1)
    Q_inverse = np.diag(np.reciprocal(np.sum(K, axis=1))) 
    #print("the inverse of Q:" , Q)

 
    # 7. form the symmetrix matrix T_hat
    T_hat = np.dot(np.dot(Q_inverse ** 0.5, K) , Q_inverse ** 0.5)
    #print(T_hat.shape)
    #print("T_hat:\n" ,T_hat)
    
    
    # 8. find the L+1 largest eigenvalues a_l and associated eigenvectors v_l of T_hat
    a_l, v_l = eigsh(T_hat,k=L+1)
    #print("a_l:\n", a_l)
    #print("v_l:\n", v_l)

    
    # 9. compute the eigenvalues of T_hat^{1/epsilon}
    eigenvalues = np.sqrt(a_l ** (1/epsilon))
    print("eigenvalues:\n", eigenvalues)
    
    
    # 10. compute the eigenvectors of the matrix T
    eigenvectors = np.dot(Q_inverse ** (0.5) ,v_l)
    print("eigenvectors:\n", eigenvectors)  
    return eigenvalues, eigenvectors
    


def plot_dataset(dataset, t_k):

    # Plotting the data set (cos(t_k),sin(t_k)) against t_k
    plt.plot(t_k, dataset[:,0]) 
    plt.plot(t_k, dataset[:,1])
    plt.xlabel('t_k')
    plt.ylabel('(cos(t_k),sin(t_k))')
    plt.grid(True)
    plt.show()
    
    # Plotting the first column of data set cos(t_k) against the second column of data set sin(t_k)
    plt.plot(dataset[:,0], dataset[:,1]) 
    plt.xlabel('cos(t_k)')
    plt.ylabel('sin(t_k)')
    plt.title('cos(t_k) against sin(t_k)')
    plt.grid(True)
    plt.show()