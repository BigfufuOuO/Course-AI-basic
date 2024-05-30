from sklearn.datasets import load_wine
import numpy as np 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


def get_kernel_function(kernel:str):
    # TODO: implement different kernel functions 
    kenerl_dict = {
        "linear": lambda x, y: np.dot(x, y),
        "poly": lambda x, y: (np.dot(x, y) + 1) ** 3,
        "rbf": lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / 2), # Gaussian kernel
        "sigmoid": lambda x, y: np.tanh(np.dot(x, y) + 1),
    }
    return kenerl_dict[kernel]

class PCA:
    def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
        self.n_components = n_components
        self.kernel_f = get_kernel_function(kernel)
        self.Matrix_reduction = None
        
    def _computerKernelMatrix(self, X:np.ndarray, kernel):
        # X: [n_samples, n_features]
        n_samples = X.shape[0]
        kenerl_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kenerl_matrix[i, j] = kernel(X[i], X[j])
        return kenerl_matrix
        pass

    def _computeKernelCentered(self, K):
        n_samples = K.shape[0]
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        return K_centered
        pass
        
    def fit(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # TODO: implement PCA algorithm
        # centralize the data
        kenerl_matrix = self._computerKernelMatrix(X, self.kernel_f)
        K_centered = self._computeKernelCentered(kenerl_matrix)
        
        eigenvalues, eigenvectors = np.linalg.eig(K_centered)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # reduction matrix
        self.Matrix_reduction = eigenvectors[:, :self.n_components]
        pass

    def transform(self, X:np.ndarray):
        # X: [n_samples, n_features]
        X_reduced = np.zeros((X.shape[0], self.n_components))
        # TODO: transform the data to low dimension
        X_reduced = np.dot(X, self.Matrix_reduction)
        return X_reduced

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # Randomly initialize the centers
    def initialize_centers(self, points):
        # points: (n_samples, n_dims,)
        n, d = points.shape

        self.centers = np.zeros((self.k, d))
        for k in range(self.k):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)
        
        return self.centers
    
    # Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        n_samples, n_dims = points.shape
        self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center

        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        pass

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Implement k-means clustering
        pass

    # Predict the closest cluster each sample in X belongs to
    def predict(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        return self.assign_points(points)
    
def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch','prince', 'ruler','princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber','guy','person','gentleman',
        'banana', 'pineapple','mango','papaya','coconut','potato','melon',
        'shanghai','HongKong','chinese','Xiamen','beijing','Guilin',
        'disease', 'infection', 'cancer', 'illness', 
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors

if __name__=='__main__':
    words, data = load_data()
    pca = PCA(n_components=2).fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=7).fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data
    
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :]) 
    plt.title("Your student ID")
    plt.savefig("PCA_KMeans.png")