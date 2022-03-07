from randomForest import *
"""
this class creates an interface compatible with other models we have written,
so that all models can be compared easily
"""

class RandomForestHyperparamters:
    def __init__(self, num_trees=70, num_features=3, max_depth=20):
        self.num_trees = num_trees
        self.num_features = num_features
        self.max_depth = max_depth


class RandomForest(RandomForestBackbone):
    def __init__(self, hyperparameters = None):
        if not hyperparameters:
            hyperparameters = RandomForestHyperparamters()
        self.forest = RandomForestBackbone(hyperparameters.num_trees, 
                                            hyperparameters.num_random_features, 
                                            hyperparameters.max_depth)
    
    def train(self, train_X, train_Y, val_X, val_Y):
        """
        train and tune a random forest
        @train_X, @train_Y: training data and labels (np arrays)
        @val_X, @val_Y: validation data and labels, will be ignored
        """
        df = pd.DataFrame(train_X.append(train_Y))
        forest.makeForest(df)
    
    def predict(self, test_X):
        """ 
        get predicted labels of testing data
        @test_X is an np array of data points
        returns np array of predictions
        """
        test_X = list(test_X)
        preds = []
        for data_point in test_X:
            preds.append(forest.classifyInstance(data_point))
        
        return np.array(preds)