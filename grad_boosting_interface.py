import typing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree
import numpy as np
import pandas as pd
from statistics import mean
import math
import warnings
from grad_booster import *
from data import *
from copy import copy


#when creating boosting interface, pass in \

class Grad_Boosting_Interface:


    def convert_data(self, y, pos_class, neg_class1, neg_class2):
        zero_count = 0
        one_count = 0
        new_y = y.copy()
        #print("the new y: " + str(new_y))

        for i in range(0, len(y)):
            if y[i] == pos_class:
                new_y[i] = 1
                one_count += 1
            else:
                if y[i] == neg_class1 or y[i] == neg_class2:
                    new_y[i] = 0
                    zero_count += 1

        #print("new y (after): " + str(new_y))
        return zero_count, one_count, new_y



    def train(self, train_X, train_y, val_X, val_y):

        self.gradBoosters = [Grad_Booster(learning_rate=0.01, boosting_rounds=5, max_depth=3, leaves=8),
                            Grad_Booster(learning_rate=0.9, boosting_rounds=1, max_depth=3, leaves=4),
                            Grad_Booster(learning_rate=0.4698, boosting_rounds=4, max_depth=3, leaves=6)]


        #now do for all boosters
        for i in range(3):
            print("i: " + str(i))
            print(self.gradBoosters[i])

            #conditional for the 3 diff classifiers
            if i == 0:
                pos_class = 0
                neg_class1 = 1
                neg_class2 = 2
            elif i == 1:
                pos_class = 1
                neg_class1 = 2
                neg_class2 = 3
            else:
                pos_class = 2
                neg_class1 = 1
                neg_class2 = 3


            train_zero_count, train_one_count, new_train_y = self.convert_data(train_y, pos_class, neg_class1, neg_class2)
            val_zero_count, val_one_count, new_val_y = self.convert_data(val_y, pos_class, neg_class1, neg_class2)
            self.gradBoosters[i].one_count = train_one_count
            self.gradBoosters[i].zero_count = train_zero_count
            self.gradBoosters[i].train(train_X, new_train_y, val_X, new_val_y)

    def predict(self, test_X):

        #number of testing points by 3
        predictions = np.zeros((len(test_X), 3))
        for i in range(3):
            pred = self.gradBoosters[i].predict(test_X)
            #convert pred back to probability if neg. value
            pred = [(math.exp(output)/(1 + math.exp(output))) for output in pred]

            predictions[:, i] = pred


            #print(predictions.shape)
        return predictions







"""train_X, train_y, val_X, val_y, test_X, test_y = get_data()
interface = Grad_Boosting_Interface()

#pass in the training data
interface.train(train_X, train_y, val_X, val_y)
print(interface.predict(test_X))"""
