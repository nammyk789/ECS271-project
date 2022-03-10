from turtle import update
import pandas as pd
import numpy as np
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
    def __init__(self, data_labels, label_set):
        counts = [data_labels.count(label) for label in label_set]
        self.probabilities = [i / len(data_labels) for i in counts]
        self.vote = max(set(data_labels), key = data_labels.count)  # get mode of list


class RandomDecisionTree:
    def __init__(self, num_random_features, max_depth=None):
        self.num_random_features= num_random_features
        self.max_depth = max_depth   # currently useless, could use later to limit depth
    
    def makeTree(self, train_data, train_labels):
        label_set = sorted(list(set(train_labels)))
        self.tree = self.decisionTreeTrain(train_data, train_labels, label_set, \
                                            self.num_random_features, self.max_depth)
    
    def getProbabilities(self, data_point, tree):
        """
        get the probability that the data point is in 
        each class
        """
        if isinstance(tree, Leaf):  # base case: we have traversed the tree
            return tree.probabilities
        else:
            if data_point[tree.feature] < tree.boundary:
                return self.getProbabilities(data_point, tree.left)
            else:
                return self.getProbabilities(data_point, tree.right)

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
        for idx, data_point in enumerate(test_data):
            if self.classifyInstance(data_point, self.tree) == test_labels[idx]:
                accuracy += 1
        return accuracy/len(test_data)
    
    def decisionTreeTrain(self, data, data_labels, label_set, num_features, max_depth, parent_impurity=1):
        """ 
        train a decision tree on inputted data
        @data: 2D array of data to train on where each array is a feature
        @data_labels: list labels of data
        @parent_impurity: impurity of the parent node
        @label_set: a list where each idx contains a data label
        """
        feature, boundary, branch_impurity = self.branchTree(data, data_labels, num_features)
        if branch_impurity > parent_impurity or max_depth == 0:
            return Leaf(data_labels, label_set)        # base case: no need to keep splitting
           
        data, data_labels = self.sortDataByFeature(data, data_labels, feature)
        split_idx = self.splitDataAtBoundary(data[feature], boundary)
        left = self.decisionTreeTrain([i[:split_idx] for i in data], data_labels[:split_idx], \
                                    label_set, num_features, max_depth - 1, branch_impurity)
        right = self.decisionTreeTrain([i[split_idx:] for i in data], data_labels[split_idx:], \
                                    label_set, num_features, max_depth - 1, branch_impurity)
        return Node(feature, boundary, left, right)
    
    
    def branchTree(self, data, data_labels, num_features):
        """
        get the best branch criteria for a random subset of features
        @data: 2D array where each row is one feature
        @data_labels: list of labels corresponding to each column in data
        @num_features: number of features to consider
        """
        rand_features = np.random.randint(low=0, high=len(data), size=num_features)
        data_series = pd.Series(data)
        rand_subset = list(data_series[rand_features])

        potential_branches = []
        for idx, feature in enumerate(rand_subset):
            branch_impurity, boundary = self.getBestBoundaryForFeature(feature, data_labels)
            potential_branches.append((rand_features[idx], boundary, branch_impurity))
        potential_branches.sort(key=lambda x:x[2])
        return potential_branches[0][0], potential_branches[0][1], potential_branches[0][2]

    def getBestBoundaryForFeature(self, feature_data, data_labels):
        """
        assuming feature data is either ranked or continuous,
        finds the decision boundary with the lowest impurity
        @feature_data: list of values for one feature
        @data_labels: labels of each data point in feature_data
        """
        feature_data, data_labels = self.sortData(feature_data, data_labels)   
        impurities = []
        boundaries = []
        for i in range(1, len(feature_data)):
            if feature_data[i - 1] < feature_data[i]:
                boundaries.append((feature_data[i - 1] + feature_data[i]) / 2)
                impurities.append(self.giniImpurity(data_labels, i))
        if impurities:
            impurities, boundaries = self.sortData(impurities, boundaries)
            return impurities[0], boundaries[0]
        else:  # case where all the data points have the same value
            return 1.0, feature_data[0]

    def giniImpurity(self, data_labels, decision_boundary_index):
        """
        calculate the Gini impurity of all the data for one feature,
        assuming data has been sorted by ascending order for that feature
        @data_labels: labels corresponding to each data point in feature
        @decision_boundary_index: index that separates data by decision boundary
        """
        impurity = 0
        left = data_labels[:decision_boundary_index]
        right = data_labels[decision_boundary_index:]
        impurity += self.giniImpurityofNode(left) * len(left) / len(data_labels)
        impurity += self.giniImpurityofNode(right) * len(right) / len(data_labels)
        return impurity

    def giniImpurityofNode(self, data_labels):
        """
        @data_labels: labels of feature data in node
        """
        impurity = 1
        for label in set(data_labels):
            num_label = len(list(filter(lambda data_label: data_label == label, data_labels)))
            impurity -= (num_label / len(data_labels))**2
        return impurity

    def sortData(self, data, data_labels):
        """
        sort data in ascending order
        @data: list of data to sort
        @data_labels: list labels of data
        """
        sorted_data =  sorted(zip(data, data_labels))
        sorted_data_points = [data_point for data_point, label in sorted_data]
        sorted_data_labels = [label for data_point, label in sorted_data]
        return sorted_data_points, sorted_data_labels

    def splitDataAtBoundary(self, feature_data, boundary):
        """
        return index of boundary for one feature
        @feature_data: array of data for one feature
        @boundary: value of boundary for that feature
        """
        for idx, val in enumerate(feature_data):
            if val > boundary:
                return idx

    def sortData(self, data, data_labels):
        """
        sort data in ascending order
        @data: list of data to sort
        @data_labels: list labels of data
        """
        sorted_data =  sorted(zip(data, data_labels))
        sorted_data_points = [data_point for data_point, label in sorted_data]
        sorted_data_labels = [label for data_point, label in sorted_data]
        return sorted_data_points, sorted_data_labels


    def sortDataByFeature(self, data, labels, feature):
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

def processDataFrame(data_frame):
    """ get feature data and data entries
    with respective labels
    @data_frame: pandas dataframe"""
    columns = data_frame.transpose().values.tolist()
    features = columns[:-1]  
    feature_labels = columns[-1]
    data = data_frame.values.tolist()
    data_labels = [i[-1] for i in data]
    data = [i[:-1] for i in data]
    return features, feature_labels, data, data_labels


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
    np.random.seed(1)
    df.iloc[np.random.permutation(len(df))]
    features, feature_labels, data, data_labels = processDataFrame(df)
    idx = 1000
    myTree = RandomDecisionTree(5, 10)
    myTree.makeTree([i[:idx] for i in features], feature_labels[:idx])
    print(myTree.classifyInstance(data[idx], myTree.tree), data_labels[idx])
    print(myTree.getProbabilities(data[idx], myTree.tree))
    # print("training accuracy:", myTree.getAccuracy(data[:idx], data_labels[:idx]))
    # print("testing accuracy:", myTree.getAccuracy(data[idx:2*idx], data_labels[idx:2*idx]))
