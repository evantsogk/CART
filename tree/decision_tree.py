import numpy as np
from .node import Node
from .splitter import gini, best_split
from .visualizer import visualize_classification_report, visualize_feature_importances


class DecisionTree:
    """A decision tree for classification based on the CART algorithm.

    Args:
        max_depth (int or None, default=None): The maximum depth of the tree. Used to control over-fitting.
        min_samples_split (int, default=2): The minimum samples for splitting a node. Used to control over-fitting.

    Attributes:
        tree (Node object): The root node of the decision tree.
        classes (ndarray): The class labels.
    """
    def __init__(self, max_depth=None, min_samples_split=2):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self.tree = None
        self.classes = None
        self._feature_importances = None
        self._n_samples = None
        self._node_id = -1

    def build(self, x, y):
        """Builds a decision tree based on the training set (x, y).

        Args:
            x (array-like, shape = [n_samples, n_features]): The training samples. Values must be numeric.
            y (array-like, shape = [n_samples]): The class labels of samples in x. Values can be integers or strings.
        """
        x, y = np.copy(x), np.copy(y)
        self.classes = np.unique(y)
        self._feature_importances = np.zeros((x.shape[1]))
        self._n_samples = x.shape[0]
        self.tree = self._build(x, y)

    def _build(self, x, y, depth=0):
        """Recursively builds a decision tree.

        Args:
            x (array-like, shape = [n_samples, n_features]): The training samples.
            y (array-like, shape = [n_samples]): The class labels of samples in x.
            depth (int): The depth of the tree.

        Returns:
            Node object: The root node of the tree.
        """
        # the current node
        class_counts = [np.sum(y == k) for k in self.classes]
        class_prediction = self.classes[np.argmax(class_counts)]
        self._node_id += 1
        node = Node(self._node_id, gini(y), x.shape[0], class_counts, class_prediction)

        # if maximum depth has been reached stop expanding
        if not (self._max_depth and depth >= self._max_depth):
            feature, threshold = best_split(x, y, self._min_samples_split)
            # feature, threshold will be None if the node cannot be split
            if feature is not None:
                node.feature, node.threshold = feature, threshold
                # the left node will get the samples with feature values <= threshold of the split
                # and the right node the rest
                permutation_left = np.nonzero(x[:, feature] <= threshold)
                permutation_right = np.nonzero(x[:, feature] > threshold)
                node.left = self._build(x[permutation_left], y[permutation_left], depth + 1)
                node.right = self._build(x[permutation_right], y[permutation_right], depth + 1)
        return node

    def predict(self, x):
        """Predicts the class of every sample in x.

        Args:
            x (array-like, shape = [n_samples, n_features]): The input samples. Values must be numeric.

        Returns:
            ndarray of shape = [n_samples]: The class predictions of samples in x.
        """
        x = np.copy(x)
        predictions = x.shape[0]*[0]
        for i, sample in enumerate(x):
            node = self.tree
            while node.left:
                node = node.left if sample[node.feature] <= node.threshold else node.right
            predictions[i] = node.class_prediction
        return np.array(predictions)

    def accuracy(self, x, y):
        """Calculates the mean accuracy on the test set (x, y).

        Args:
            x (array-like, shape = [n_samples, n_features]): The test samples. Values must be numeric.
            y (array-like, shape = [n_samples]): The test class labels. Values can be integers or strings.

        Returns:
            float: The mean accuracy.
        """
        predictions = self.predict(x)
        return np.mean(predictions == y)

    def classification_report(self, x, y, plot=False, cmap='YlOrRd'):
        """The classification report consists of the precision, the recall and the F1 scores. It returns a dictionary
        with the scores and will also display a heatmap if the argument plot is set to True.

        Args:
            x (array-like, shape = [n_samples, n_features]): The test samples. Values must be numeric.
            y (array-like, shape = [n_samples]): The test class labels. Values can be integers or strings.
            plot (bool, default=False): If true, displays heatmap.
            cmap (string, default='YlOrRd'): The name of the colormap to be used for the heatmap if plot=True.

        Returns:
            dict: A dictionary that contains the precision, recall and F1 scores for each class.
        """
        y = np.copy(y)
        predictions = self.predict(x)
        report = {}
        for c in self.classes:
            tp = sum(np.sum(predictions[i] == c and y[i] == c) for i in range(y.shape[0]))
            precision = tp / np.sum(predictions == c)
            recall = tp / np.sum(y == c)
            f1_score = 2*precision*recall / (precision + recall)
            report[c] = {'precision': precision, 'recall': recall, 'f1': f1_score}
        if plot:
            visualize_classification_report(report, self.classes, cmap)
        return report

    def feature_importances(self, plot=False, feature_names=None, color='steelblue'):
        """Returns the feature importances. If the argument plot is set to true it will also display them in a plot.

        The feature importance is computed as the normalized total reduction of node gini impurity weighted by the
        probability of reaching that node.

        Args:
            plot (bool, default=False): If true, displays the feature importances in a plot.
            feature_names (array-like, shape = [n_features]): The feature names. Used for the plot.
            color (string, default='steelblue'): The color to use for the plot.

        Returns:
            ndarray of shape = [n_features]: The feature importances.
        """
        self._calc_feature_importances(self.tree)
        # normalize feature importances to be summed to 1
        self._feature_importances = self._feature_importances / np.sum(self._feature_importances)
        if plot:
            visualize_feature_importances(self._feature_importances, feature_names, color)
        return self._feature_importances

    def _calc_feature_importances(self, node):
        """Calculates the feature importances.

        Args:
            node (Node object): The current node. The initial node is the root of the tree.
        """
        if node.left:
            node_importance = node.samples / self._n_samples * node.gini - \
                              node.left.samples / self._n_samples * node.left.gini - \
                              node.right.samples / self._n_samples * node.right.gini
            self._feature_importances[node.feature] += node_importance

            self._calc_feature_importances(node.left)
            self._calc_feature_importances(node.right)

    def get_depth(self):
        """Returns the depth of the decision tree.
        """
        return self._get_depth(self.tree)

    def _get_depth(self, node, current_depth=0, depth=0):
        """Recursively finds the depth of the decision tree.

        Args:
            node (Node object): The current node. The initial node is the root of the tree.
            current_depth (int): The depth of the current node.
            depth (int): The maximum depth found.
        """
        if depth < current_depth:
            depth = current_depth
        if node.left:
            depth = self._get_depth(node.left, current_depth + 1, depth)
            depth = self._get_depth(node.right, current_depth + 1, depth)
        return depth

    def get_n_leaves(self):
        """Returns the number of leaves of the decision tree.
        """
        return self._get_n_leaves(self.tree)

    def _get_n_leaves(self, node, leaves=0):
        """Recursively finds the number of leaves of the decision tree.

        Args:
            node (Node object): The current node. The initial node is the root of the tree.
            leaves (int): The number of leaves.
        """
        if node.left is None:
            leaves += 1
        else:
            leaves = self._get_n_leaves(node.left, leaves)
            leaves = self._get_n_leaves(node.right, leaves)
        return leaves

    def print_tree(self, max_depth=None):
        """Prints the information of the tree's nodes.

        Args:
            max_depth (int or None, default=None): The max depth to print.
        """
        self._print_tree(self.tree, max_depth)

    def _print_tree(self, node, max_depth, depth=0):
        """Recursively prints the information of the tree's nodes.

        Args:
            node (Node object): The current node. The initial node is the root of the tree.
            max_depth (int or None): The max depth to print.
            depth (int): The current depth.
        """
        if max_depth and depth > max_depth:
            return
        print("Depth:", depth)
        if node.left is None:
            print("node #" + str(node.node_id), "| gini =", "%.3f" % round(node.gini, 3), "| samples =", node.samples,
                  "| value =", node.class_counts, "| class =", node.class_prediction)
        else:
            print("node #" + str(node.node_id), "| X" + str(node.feature), "<=", node.threshold,
                  "| gini =", "%.3f" % round(node.gini, 3), "| samples =", node.samples, "| value =", node.class_counts,
                  "| class =", node.class_prediction)
            self._print_tree(node.left, max_depth, depth + 1)
            self._print_tree(node.right, max_depth, depth + 1)
