import pandas as pd


class BinaryTreeNode:
    def __init__(self, impurity):
        self.impurity = impurity
        self.left = None
        self.right = None



def branchTree(data, data_labels):
    """
    get the best branch criteria for a set of features
    @data: 2D array where each row is one feature
    @data_labels: list of labels corresponding to each column in data
    """
    potential_branches = []
    for idx, feature in enumerate(data):
        branch_impurity, boundary = getBestBoundaryForFeature(feature, data_labels)
        potential_branches.append((idx, boundary, branch_impurity))
    potential_branches.sort(key=lambda x:x[2])
    return potential_branches[0][0], potential_branches[0][1], potential_branches[0][2]


def getBestBoundaryForFeature(feature_data, data_labels):
    """
    assuming feature data is either ranked or continuous,
    finds the best decision boundary
    @feature_data: list of values for one feature
    @data_labels: labels of each data point in feature_data
    """
    feature_data, data_labels = sortData(feature_data, data_labels)   
    impurities = []
    boundaries = []
    for i in range(1, len(feature_data)):
        if feature_data[i - 1] < feature_data[i]:
            boundaries.append((feature_data[i - 1] + feature_data[i]) / 2)
            impurities.append(giniImpurity(data_labels, i))
    impurities, boundaries = sortData(impurities, boundaries)
    return impurities[0], boundaries[0]


def giniImpurity(data_labels, decision_boundary_index):
    """
    calculate the Gini impurity of all the data for one feature,
    assuming data has been sorted by ascending order for that feature
    @data_labels: labels corresponding to each data point in feature
    @decision_boundary_index: index that separates data by decision boundary
    """
    impurity = 0
    left = data_labels[:decision_boundary_index]
    right = data_labels[decision_boundary_index:]
    impurity += giniImpurityofNode(left) * len(left) / len(data_labels)
    impurity += giniImpurityofNode(right) * len(right) / len(data_labels)
    return impurity


def giniImpurityofNode(data_labels):
    """
    @data_labels: labels of feature data in node
    """
    impurity = 1
    for label in set(data_labels):
        num_label = len(list(filter(lambda data_label: data_label == label, data_labels)))
        impurity -= (num_label / len(data_labels))**2
    return impurity


def sortData(data, data_labels):
    """
    sort data in ascending order
    """
    sorted_data =  sorted(zip(data, data_labels))
    sorted_data_points = [data_point for data_point, label in sorted_data]
    sorted_data_labels = [label for data_point, label in sorted_data]
    return sorted_data_points, sorted_data_labels


# def getSubset(data, labels, target_label):
#     """
#     get subset of data that corresponds to a class
#     @data: all data
#     @labels: labels for data
#     @target_label: class we want to filter for"""
#     subset = []    # might have to edit this to drop NULL values
#     for idx, data_point in enumerate(data):
#         if labels[idx] == target_label:
#             subset.append(data_point)
#     return subset


if __name__ == "__main__":
    # test_column = [155, 180, 190, 220, 225]
    # test_labels = [0, 1, 0, 1, 1]
    # print("test column:", test_column)
    # print("test labels:", test_labels)
    # test_column, test_labels = sortData(test_column, test_labels)
    # print("impurity:", giniImpurity(test_labels, 1))
    # should be (0.27, 205)
    # print("best boundary:", getBestBoundaryForFeature(test_column, test_labels))
    df = pd.read_csv("fetal_health.csv")
    columns = df.transpose().values.tolist()
    data = columns[:-1]
    labels = columns[-1]
    print(branchTree(data, labels))
