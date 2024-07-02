from sklearn.datasets import load_wine
import numpy as np 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from Debug import PicPoints


def get_kernel_function(kernel:str):
    # TODO: implement different kernel functions 
    kenerl_dict = {
        "linear": lambda x, y: np.dot(x, y),
        "poly": lambda x, y: (np.dot(x, y) + 1) ** 3,
        "rbf": lambda x, y: np.exp(-np.linalg.norm(x - y) ** 2 / 100), # Gaussian kernel
        "laplacian": lambda x, y: np.exp(-np.linalg.norm(x - y) / 4), # Laplacian kernel
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
        draw = PicPoints().to_mtrx(kenerl_matrix)
        K_centered = self._computeKernelCentered(kenerl_matrix)
        
        eigenvalues, eigenvectors = np.linalg.eig(K_centered)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:self.n_components]
        eigenvectors = eigenvectors[:, idx][:, :self.n_components]
        
        # reduction matrix
        return eigenvalues, eigenvectors, kenerl_matrix
        pass

    def transform(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # TODO: transform the data to low dimension
        V, A, K = self.fit(X)
        # normalize the A
        # A = A / np.linalg.norm(A, axis=1, keepdims=True)
        X_reduced = K @ A / np.sqrt(V)
        return X_reduced

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None
        self.k = n_clusters

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
        for i in range(n_samples):
            self.labels[i] = np.argmin(np.linalg.norm(self.centers - points[i], axis=1)) # L2 norm
        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        for k in range(self.k):
            clusters = points[self.labels == k]
            if len(clusters) > 0:
                self.centers[k] = clusters.mean(axis=0)
        pass

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Implement k-means clustering
        while self.max_iter > 0:
            old = self.centers.copy()
            self.assign_points(points)
            self.update_centers(points)
            self.max_iter -= 1
            if np.all(old == self.centers):
                break
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
    w2v = KeyedVectors.load_word2vec_format('./part_1/data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors

if __name__=='__main__':
    words, data = load_data()
    pca = PCA(n_components=2, kernel="rbf")
    pca.fit(data)
    data_pca = pca.transform(data)
    
    dbg = PicPoints()
    dbg.plot(data_pca, words)

    kmeans = KMeans(n_clusters=7, max_iter=10)
    kmeans.initialize_centers(data_pca)
    kmeans.fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data
    
    plt.figure(figsize=(10, 10))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :]) 
    plt.title("PB21061361")
    plt.savefig("PCA_KMeans.png")