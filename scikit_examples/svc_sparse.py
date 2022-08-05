import numpy as np
from sklearnex import patch_sklearn
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand


def check_sparse_support(X, sparse_matrix, non_sparse_matrix, sparse_model, non_sparse_model, y=None):
  # Fit with a sparse matrix
  try:
    if y is not None:
      sparse_model.fit(X, y)
    else:
      sparse_model.fit(X)
    print("Sparse training success")
  except:
    print("Sparse training failure")

  # Fit with a non sparse matrix
  try:
    if y is not None:
      non_sparse_model.fit(X.toarray(), y)
    else:
      non_sparse_model.fit(X.toarray())
    print("Non sparse training success")
  except:
    print("Non sparse training failure")

  # Predict with a sparse matrix
  try:
    sparse_model.predict(sparse_matrix)
    print("Sparse inference success")
  except:
    print("Sparse inference failure")

  # Predict with a non sparse matrix
  try:
    non_sparse_model.predict(non_sparse_matrix)
    print("Non sparse inference success")
  except:
    print("Non sparse inference failure")


if __name__ == "__main__":
  num_samples = 1000
  num_features = 1000

  # Initialize sparse matrix for training, randomly set some values
  X = lil_matrix((num_samples, num_features))
  X[0, :100] = rand(100)
  X[1, 100:200] = X[0, :100]
  X.setdiag(rand(1000))
  X = X.tocsr()

  # Intialize sparse and non-sparse matricies for inference
  sparse_matrix = lil_matrix((1000, 1000))
  sparse_matrix.setdiag(rand(1000))
  sparse_matrix = sparse_matrix.tocsr()
  non_sparse_matrix = sparse_matrix.toarray()

  patch_sklearn()

  # Linear regression
  from sklearn.linear_model import LinearRegression
  print("\nChecking Linear Regression...")
  check_sparse_support(X, sparse_matrix, non_sparse_matrix, LinearRegression(), LinearRegression(), y=rand(num_samples))

  # K means
  from sklearn.cluster import KMeans
  print("\nChecking K Means...")
  check_sparse_support(X, sparse_matrix, non_sparse_matrix, KMeans(), KMeans())

  # SVC
  from sklearn.svm import SVC
  print("\nChecking Support Vector Classifier...")
  check_sparse_support(X, sparse_matrix, non_sparse_matrix, SVC(gamma='auto'), SVC(gamma='auto'), y=np.random.choice(2, num_samples))

