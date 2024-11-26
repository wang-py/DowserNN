import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    input_data_X = np.load(sys.argv[1])

    # components in input data
    components = 70
    pca = PCA(n_components=components)
    X_scaled = StandardScaler().fit_transform(input_data_X)
    pca_features = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(pca_features[0, :], pca_features[1, :])
    ax[1].hist(pca_features[:, 0])
    plt.show()

    pass
