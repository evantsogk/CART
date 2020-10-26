import numpy as np
import matplotlib.pyplot as plt


def visualize_classification_report(report, classes, cmap):
    """Creates a heatmap for the precision, recall and f1 scores of the classes.

    Args:
        report (dict): A dictionary with the precision, recall and f1 scores of each class.
        classes (array): The class labels.
        cmap: (string): The name of the colormap to be used for the heatmap.
    """
    values = np.array([[report[c]['precision'], report[c]['recall'], report[c]['f1']] for c in report])

    # create the heatmap
    fig, ax = plt.subplots()
    pcolor = ax.pcolor(values, edgecolors='white', linewidths=0.8, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5)
    ax.set_xticklabels(['Precision', 'Recall', 'F1'])
    ax.set_yticklabels(classes)

    # add values to the heatmap
    pcolor.update_scalarmappable()
    for path, color, value in zip(pcolor.get_paths(), pcolor.get_facecolors(), pcolor.get_array()):
        x, y = path.vertices[:-2, :].mean(0)
        # set text color according to background color for better visibility
        color = (0.0, 0.0, 0.0) if np.all(color[:3] > 0.5) else (1.0, 1.0, 1.0)
        ax.text(x, y, "%.3f" % value, ha="center", va="center", color=color)

    cbar = plt.colorbar(pcolor, values=np.arange(0.0, 1.1, 0.1))
    cbar.outline.set_edgecolor('lightgray')
    cbar.outline.set_linewidth(1.2)
    plt.setp(ax.spines.values(), color='silver')
    plt.title("Classification Report")


def visualize_feature_importances(feature_importances, feature_names, color):
    """Creates a bar chart for the feature importances.

    Args:
        feature_importances (array-like, shape = [n_features]): The feature importances.
        feature_names (array-like, shape = [n_features]): The feature names.
        color (string): The color of the bars.
    """
    n_features = feature_importances.shape[0]
    sorted_indices = np.argsort(feature_importances)

    fig, ax = plt.subplots()
    plt.barh(range(n_features), feature_importances[sorted_indices], color=color)
    if feature_names is None:
        plt.yticks(range(n_features), sorted_indices)
    else:
        plt.yticks(range(n_features), feature_names)

    ax.xaxis.grid(color='lightgray')
    ax.set_axisbelow(True)
    plt.setp(ax.spines.values(), color='lightgray')
    plt.title("Feature Importances")
