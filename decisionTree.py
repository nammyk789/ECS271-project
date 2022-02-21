import pandas as pd
"""
to-do: implement depth control, maybe implement some impurity threshhold parameter
to prevent overfitting
"""

class Node:
    def __init__(self, feature, boundary, left, right):
        self.feature = feature
        self.boundary = boundary
        self.left = left
        self.right = right


class Leaf:
    def __init__(self, data_labels):
        self.vote = max(set(data_labels), key = data_labels.count)  # get mode of list


class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth   # need to implement, for controlling depth
    
    def makeTree(self, train_data, train_labels):
        self.tree = decisionTreeTrain(train_data, train_labels)
    
    def classifyInstance(self, data_point, tree):
        """
        get the decision tree's classification of a data point
        @data_point: data point to classify
        """
        if isinstance(tree, Leaf):  # base case: we have traversed the tree
            return tree.vote
        else:
            if data_point[tree.feature] < tree.boundary:
                return self.classifyInstance(data_point, tree.left)
            else:
                return self.classifyInstance(data_point, tree.right)
    
    def getAccuracy(self, test_data, test_labels):
        """
        get the accuracy of the decision tree, as a percentage of correctly
        classified data
        @test_data: testing data points
        @test_labels: labels of testing data points
        """
        accuracy = 0
        for idx in range(len(test_data[0])):
            if self.classifyInstance([i[idx] for i in test_data], self.tree) == test_labels[idx]:
                accuracy += 1
        return accuracy/len(test_data[0])



def decisionTreeTrain(data, data_labels, parent_impurity=1):
    """ 
    train a decision tree on inputted data
    @data: 2D array of data to train on
    @data_labels: list labels of data
    @parent_impurity: impurity of the parent node
    """
    feature, boundary, branch_impurity = branchTree(data, data_labels)
    if branch_impurity > parent_impurity:
        return Leaf(data_labels)        # base case: no need to keep splitting
    else:
        data, data_labels = sortDataByFeature(data, data_labels, feature)
        split_idx = splitDataAtBoundary(data[feature], boundary)
        left = decisionTreeTrain([i[:split_idx] for i in data], data_labels[:split_idx], branch_impurity)
        right = decisionTreeTrain([i[split_idx:] for i in data], data_labels[split_idx:], branch_impurity)
        return Node(feature, boundary, left, right)



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
    finds the decision boundary with the lowest impurity
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
    if impurities:
        impurities, boundaries = sortData(impurities, boundaries)
        return impurities[0], boundaries[0]
    else:  # case where all the data points have the same value
        return 1.0, feature_data[0]


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
    @data: list of data to sort
    @data_labels: list labels of data
    """
    sorted_data =  sorted(zip(data, data_labels))
    sorted_data_points = [data_point for data_point, label in sorted_data]
    sorted_data_labels = [label for data_point, label in sorted_data]
    return sorted_data_points, sorted_data_labels


def splitDataAtBoundary(feature_data, boundary):
    """
    return index of boundary for one feature
    @feature_data: array of data for one feature
    @boundary: value of boundary for that feature
    """
    for idx, val in enumerate(feature_data):
        if val > boundary:
            return idx


def sortData(data, data_labels):
    """
    sort data in ascending order
    @data: list of data to sort
    @data_labels: list labels of data
    """
    sorted_data =  sorted(zip(data, data_labels))
    sorted_data_points = [data_point for data_point, label in sorted_data]
    sorted_data_labels = [label for data_point, label in sorted_data]
    return sorted_data_points, sorted_data_labels


def sortDataByFeature(data, labels, feature):
    """
    sort all data and labels based on one feature column
    https://stackoverflow.com/questions/2173797/how-to-sort-2d-array-by-row-in-python
    @data: 2D array of data
    @labels: labels of data
    @feature: index of feature in data
    """
    data_transpose = sorted(zip(*data, labels), key=lambda x: x[feature])
    data_and_labels = zip(*data_transpose)
    data_and_labels = list(map(list, data_and_labels))  # convert back from list of tuples to list of lists
    data = data_and_labels[:-1]
    labels = data_and_labels[-1]
    return data, labels


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
    idx = 1000
    myTree = DecisionTree()
    myTree.makeTree([i[:idx] for i in data], labels[:idx])
    # print(myTree.classifyInstance([i[idx] for i in data], myTree.tree))
    print(myTree.getAccuracy([i[:idx] for i in data], labels[:idx]))
    print(myTree.getAccuracy([i[idx:2*idx] for i in data], labels[idx:2*idx]))
