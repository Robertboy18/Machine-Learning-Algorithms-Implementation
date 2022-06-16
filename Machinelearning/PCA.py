import numpy as np
import matplotlib.pyplot as plt

def pca(X):
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

def recover_data(Z, U, k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

def main():
    # load the data
    X = np.loadtxt('data/ex7data1.txt', delimiter=',')

    # plot the data
    plt.scatter(X[:, 0], X[:, 1], marker='o', c='b', edgecolor='k', s=40)
    plt.title("Example Dataset 1")
    plt.show()

    # run PCA
    U, S, V = pca(X)

    # draw the eigenvectors centered at mean of data
    plt.scatter(X[:, 0], X[:, 1], marker='o', c='b', edgecolor='k', s=40)
    plt.title("Example Dataset 1")
    plt.plot([X.mean(), U[0,0] + X.mean()], [X.mean(), U[0,1] + X.mean()], c='k', linewidth=3)
    plt.plot([X.mean(), U[1,0] + X.mean()], [X.mean(), U[1,1] + X.mean()], c='k', linewidth=3)
    plt.show()

    # project the data onto the principal components
    Z = project_data(X, U, 1)

    # recover the data
    X_recovered = recover_data(Z, U, 1)

    # plot the data
    plt.scatter(X_recovered[:, 0], X_recovered[:, 1], marker='o', c='b', edgecolor='k', s=40)
    plt.title("Recovered data")
    plt.show()

if __name__ == '__main__':
    main()
