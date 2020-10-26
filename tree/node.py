class Node:
    """Tree node representation.

    Args:
        node_id (int): The unique id of the node.
        gini (float): The gini impurity of the node.
        samples (int): The number of samples of the node.
        class_counts (array, shape[n_classes]): The number of samples corresponding to each class.
        class_prediction (int or string): The predicted class of the node, which is the class with the most samples.
        feature (int): The feature index of the split.
        threshold (float): The threshold value of the split.
        left (Node object): The left child of the node.
        right (Node object): The right child of the node.
    """
    def __init__(self, node_id, gini, samples, class_counts, class_prediction, feature=None, threshold=None, left=None,
                 right=None):
        self.node_id = node_id
        self.gini = gini
        self.samples = samples
        self.class_counts = class_counts
        self.class_prediction = class_prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
