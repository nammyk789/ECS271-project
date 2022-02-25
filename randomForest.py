from random import Random
from decisionTree import *

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
class RandomForest(RandomDecisionTree):
    def __init__(self, num_trees, num_random_features):
        self.num_trees = num_trees
        self.num_random_features = num_random_features
    
    def classifyInstance(self, data_point):
        """
        classify data point
        """
        votes = []
        for tree in self.trees:
            votes.append(tree.classifyInstance(data_point))
        return max(set(votes), key = votes.count)
    
    def testOutOfBag(self, data, labels):
        """
        test how many of the out of bag data points
        are correctly classified
        @data: 2D matrix of unlabeled data points
        @labels: list of labels for data"""
        correctly_classified = 0
        for idx in self.out_of_bag_idx:
            if self.classifyInstance(data[idx]) == labels[idx]:
                correctly_classified += 1
        return correctly_classified / len(self.out_of_bag_idx)
    
    def makeForest(self, data, labels):
        """
        generate a random forest
        @data: 2D matrix of unlabeled data points
        @labels: list of labels for data        """
        self.trees = []
        self.out_of_bag_idx = set() # keep track of out of bag data
        data_and_labels = data
        data_and_labels.append(labels)
        for i in range(self.num_trees):
            bootstrapped_data, out_of_bag = generateBootstrappedData(data_and_labels)
            data = bootstrapped_data[:-1]
            labels = bootstrapped_data[-1]
            tree = RandomDecisionTree(self.num_random_features)
            tree = tree.makeTree(data, labels)
            self.trees.append[tree]
            self.out_of_bag_idx.add(*out_of_bag)



    

def generateBootstrappedData(data):
    """
    get randomly sampled dataset of the same size
    @data: 2D array where each entry is a row in data table
    """
    rand_rows = np.random.randint(low=0, high=len(data), size=len(data))
    data_series = pd.Series(data)
    bootstrapped_data = list(data_series[rand_rows])
    out_of_bag = []
    for i in range(len(data)):
        if i not in rand_rows:
            out_of_bag.append(i)
    return bootstrapped_data, out_of_bag


if __name__ == "__main__":
    df = pd.read_csv("fetal_health.csv")
    np.random.seed(1)
    df.iloc[np.random.permutation(len(df))]
    columns = df.transpose().values.tolist()
    data = columns[:-1]
    labels = columns[-1]
    forest = RandomForest(50, 3)
    forest.makeForest(data, labels)
    # print(forest.train)