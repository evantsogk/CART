import numpy as np


def gini(y):
    """Calculates the gini impurity of a node.

    Args:
        y (array-like, shape[n_samples]): The class labels of a node's samples.

    Returns:
        float: The gini impurity.
    """
    classes, counts = np.unique(y, return_counts=True)
    return 1 - np.sum((counts / y.shape[0]) ** 2)


def best_split(x, y, min_samples_split):
    """Finds the best split point of a node.

    Checks all the possible split points for every feature and returns the feature index and threshold value of the
    split that minimizes the gini impurity.

    Args:
        x (array-like, shape = [n_samples, n_features]): The node's samples.
        y (array-like, shape = [n_samples]): The node's class labels.
        min_samples_split (int): The minimum samples a node must have to consider splitting it.

    Returns:
        best_feature (int): The feature index of the best split.
        best_threshold (float): The threshold value of the best split.
    """
    n = y.shape[0]
    best_feature, best_threshold = None, None
    best_gini = gini(y)

    # stop splitting if the node has less than the minimum samples or if all the samples belong to the same class
    if n < min_samples_split or best_gini == 0:
        return best_feature, best_threshold

    for feature in range(x.shape[1]):
        permutation = x[:, feature].argsort()
        x_sorted, y_sorted = x[permutation], y[permutation]

        # can't split between same values so skip
        for i in range(1, n):
            if x_sorted[i, feature] == x_sorted[i - 1, feature]:
                continue

            # check if the split impurity is minimum
            gini_split = (i * gini(y_sorted[:i]) + (n - i) * gini(y_sorted[i:])) / n
            if gini_split < best_gini:
                best_gini = gini_split
                best_feature = feature
                best_threshold = (x_sorted[i, feature] + x_sorted[i - 1, feature]) / 2
    return best_feature, best_threshold
