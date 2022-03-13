import time
import numpy as np
import pandas as pd
from models.random_forest.decision_tree import *

"""
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
method:
- randomly sample from data (repeating samples is allowed) until you have a dataset
of the same size
- generate a decision tree, but at each step use a random subset of the features
- do this 100s of times
- for classification, take a vote amongst decision trees
- out of bag samples: samples that were not included in a decision tree
- accuracy: how many out of bag samples are correctly classified by the random forest
- hyperparameter tuning: changing the number of random features we use per branch. start with
the square root of the number of variables
"""
class RandomForestBackbone(RandomDecisionTree):
    def __init__(self, num_trees, num_random_features, max_depth):
        self.num_trees = num_trees
        self.num_random_features = num_random_features
        self.max_depth = max_depth
    
    def getInstanceProbabilities(self, data_point):
        """
        get the probability that data point belongs to each class
        """
        probabilities = []
        for tree in self.trees:
            probabilities.append(tree.getProbabilities(data_point, tree.tree))
        prob_array = np.array(probabilities)
        return prob_array.mean(axis=0)
        

    def classifyInstance(self, data_point):
        """
        classify data point
        """
        votes = []
        for tree in self.trees:
            votes.append(tree.classifyInstance(data_point, tree.tree))
        return max(set(votes), key = votes.count)
    
    def getAccuracy(self, dataframe):
        """
        get the accuracy of the random forest, as a percentage of correctly
        classified data
        @data: testing data points
        @labels: labels of testing data points
        """
        data, labels = processDataFrame(dataframe)[2:]
        accuracy = 0
        for idx, data_point in enumerate(data):
            if self.classifyInstance(data_point) == labels[idx]:
                accuracy += 1
        return accuracy / len(data)
    
    def testOutOfBag(self, dataframe):
        """
        test how many of the out of bag data points
        are correctly classified
        @data: 2D matrix of unlabeled data points
        @labels: list of labels for data
        """
        data, labels = processDataFrame(dataframe)[2:]
        correctly_classified = 0
        for idx in self.out_of_bag_idx:
            if self.classifyInstance(data[idx]) == labels[idx]:
                correctly_classified += 1
        return correctly_classified / len(self.out_of_bag_idx)
    
    def makeForest(self, data_frame, verbose = False):
        """
        generate a random forest
        """
        start_time = time.time()
        self.trees = []
        out_of_bag_idx = [] # keep track of out of bag data
        for _ in range(self.num_trees):
            bootstrapped_data_frame, out_of_bag = generateBootstrappedData(data_frame)
            features, labels = processDataFrame(bootstrapped_data_frame)[:2]
            tree = RandomDecisionTree(self.num_random_features, max_depth=self.max_depth) 
            tree.makeTree(features, labels)
            self.trees.append(tree) 
            out_of_bag_idx.extend(out_of_bag)
        self.out_of_bag_idx = set(out_of_bag_idx)
        if verbose:
            print("Finished training! Time elapsed:", time.time() - start_time)
    

def generateBootstrappedData(data_frame):
    """
    get randomly sampled dataset of the same size
    @data_frame: pandas dataframe
    """
    rand_rows = np.random.randint(low=0, high=len(data_frame), size=len(data_frame))
    bootstrapped_data_frame = data_frame.iloc[rand_rows]
    out_of_bag = []
    for i in range(len(data_frame)):
        if i not in rand_rows:
            out_of_bag.append(i)
    return bootstrapped_data_frame, out_of_bag 


if __name__ == "__main__":
    df = pd.read_csv("fetal_health.csv")
    np.random.seed(1)
    df.iloc[np.random.permutation(len(df))]
    data, data_labels = processDataFrame(df)[2:]
    forest = RandomForestBackbone(50, 3, 100)
    forest.makeForest(df)
    print("classifier:", forest.classifyInstance(data[1]), "real label:", data_labels[1])
    print("probabilities:", forest.getInstanceProbabilities(data[1]))
    print("out of bag accuracy:", forest.testOutOfBag(df))
    print("testing accuracy:", forest.getAccuracy(df))
