from random_forest import *
import pandas as pd
"""
this class creates an interface compatible with other models we have written,
so that all models can be compared easily
"""

class RandomForestHyperparamters:
    def __init__(self, num_trees=143, num_features=9, max_depth=11):
        self.num_trees = num_trees
        self.num_features = num_features
        self.max_depth = max_depth


class RandomForest(RandomForestBackbone):
    def __init__(self, hyperparameters = None):
        if not hyperparameters:
            hyperparameters = RandomForestHyperparamters()
        self.forest = RandomForestBackbone(hyperparameters.num_trees, 
                                            hyperparameters.num_features, 
                                            hyperparameters.max_depth)
    
    def train(self, train_X, train_y, val_X, val_y):
        """
        train and tune a random forest
        @train_X, @train_Y: training data and labels (np arrays)
        @val_X, @val_Y: validation data and labels, will be ignored
        """
        self.forest.makeForest(self.makeAllData(train_X, train_y))
    
    def predict(self, test_X):
        """ 
        get predicted labels of testing data
        @test_X is an np array of data points
        returns np array of predictions
        """
        test_X = list(test_X)
        preds = []
        for data_point in test_X:
            preds.append(self.forest.getInstanceProbabilities(data_point))
        return np.array(preds)
    
    def accuracy(self, test_X, test_y):
        return self.forest.getAccuracy(self.makeAllData(test_X, test_y))
    
    def makeAllData(self, data_X, data_y):
        """
        take np arrays of data and labels and make a pandas df
        """
        all_data = np.insert(data_X, data_X.shape[1], data_y, axis=1)
        return pd.DataFrame(all_data, columns = [i for i in range(all_data.shape[1])])