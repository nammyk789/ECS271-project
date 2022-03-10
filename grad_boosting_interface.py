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



class Grad_Boosting_Interface:


    def __init__(self, one_count, zero_count, model, learning_rate, boosting_rounds):
        self.one_count = one_count
        self.zero_count = zero_count
        self.first_tree = True
        self.new_probs = []
        self.first_pred = 0
        self.new_pseudo_resids = []
        self.i = 0
        self.loss_list = []
        self.curr_pred = 0
        self.model = model
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds

    #@abstractmethod
    def train(self, train_X: np.array, train_y: np.array, val_X: np.array, val_y: np.array) -> None:
        """
        Train the classifier using training data
        @train_X: Numpy array of shape (N, d,) consisting of the training features
        @train_y: Numpy array of shape (N,) consisting of the training target labels
        @val_X: Numpy array of shape (N, d,) consisting of the validation features
        @val_y: Numpy array of shape (N,) consisting of the validation target labels

        Calculate log(odds) of normal fetus and then pass into logistic function.
        Plug into logistic regression equation.
        Calculate residuals (pseudo residuals) for each class - y_train - in our data

        Fit the decisiontree regressor to the x_train data and pseudo residuals. We have 21 features in this data
        Predict the probabilities from using new weak learner and its leaves.
        Get the new pseudo residuals

        Repeat for all boosting rounds and calculate training accuracy.
        """


        log_odds = math.log(self.one_count/self.zero_count)
        print("log odds: " + str(log_odds))

        log_regression = math.exp(log_odds)/(1 + math.exp(log_odds))

        self.first_pred = log_regression
        print("prediction for first boosting round: " + str(self.first_pred))

        self.new_pseudo_resids = [0]*len(train_y)
        count = 0

        for elem in range(0, len(train_y)):
            self.new_pseudo_resids[count] = train_y[elem] - self.first_pred
            count += 1


        self.model = self.model.fit(train_X, self.new_pseudo_resids)
        print(self.model)
        #tree.plot_tree(self.model)
        #plt.show()

        leaf_output_list = self.apply_residual_transformation(train_X)

        self.predict_prob(train_X, train_y, leaf_output_list)


        self.new_pseudo_resids = [0]*len(train_y)

        for i in range(0, len(train_y)):
            self.new_pseudo_resids[i] = train_y[i] - self.new_probs[i]

        self.loss_list.append(sum(self.new_pseudo_resids))

        self.calc_accuracy(train_X, train_y, leaf_output_list)

        #plotting the final tree
        #tree.plot_tree(self.model)
        #plt.show()
        print("the loss list: " + str(self.loss_list))
        print()





    def calc_accuracy(self, train_X, train_y, leaf_output_list):

        """
        @train_X: Numpy array of shape (N, d,) consisting of the training features
        @train_y: Numpy array of shape (N,) consisting of the training target labels
        @leaf_output_list: tuple list of the new leaves and their outputs and calculated from the new pseudo residual calculation.

        Repeats algorithm from train() and Calculates accuracy from each boosting round (equivalent to epoch) on the training set.
        """

        curr_acc = []
        total_preds = len(train_y)

        #repeat
        for i in range(0, self.boosting_rounds):

            print()
            print("boosting round: " + str(2 + i))

            self.model = self.model.fit(train_X, self.new_pseudo_resids)
            leaf_output_list = self.apply_residual_transformation(train_X)
            self.predict_prob(train_X, train_y, leaf_output_list)

            correct_preds = 0
            for j in range(0, len(train_y)):
                if self.new_probs[j] >= 0.5 and train_y[j] == 1:
                    correct_preds += 1

                else:
                    if self.new_probs[j] < 0.5 and train_y[j] == 0:
                        correct_preds += 1

                self.new_pseudo_resids[j] = train_y[j] - self.new_probs[j]

            curr_acc = (correct_preds/total_preds)*100
            print("the current training accuracy for epoch " + str(i) + ": " + str(curr_acc))
            print()
            self.loss_list.append(sum(self.new_pseudo_resids))





    def calc_leaf_output(self, leaf_indices, leaf_output_dict, leaf_output_list, unique_leaves):

        """
        @leaf_indices: The leaf indices of all the leaves mapped every training point.
        @leaf_output_dict: Dictionary storing the leaves along with the pseudo residuals that are mapped to them.
        @leaf_output_list: tuple list of the new leaves and their outputs and calculated from the new pseudo residual calculation.
        @unique_leaves: The number of unique leaves in DecisionTreeRegressor.
        returns: new leaf output list

        Calculates leaf output list for all boosting rounds besides the first.
        """

        curr_numerator = 0
        curr_denominator = 0

        prev_prob = 0

        for i in range(0, len(unique_leaves)):
            curr_leaf = unique_leaves[i]

            curr_numerator = sum(j for i, j in leaf_output_dict[curr_leaf])
            curr_leaf_indices = [x for x in leaf_indices if x == curr_leaf]


            for j in range(0, len(curr_leaf_indices)):
                train_leaf = curr_leaf_indices[j]
                index = leaf_output_dict[train_leaf][j][0]

                prev_prob = self.new_probs[index]
                curr_denominator += prev_prob * (1 - prev_prob)

            output = curr_numerator/curr_denominator
            leaf_output_list.append((curr_leaf, output))


        return leaf_output_list




    #note: can use model.apply to see where leaf maps to
    #apply the transformation of the residual since we're doing a classification
    def apply_residual_transformation(self, train_X):
        """
        @train_X: Numpy array of shape (N, d,) consisting of the training features
        returns: new leaf output list

        Get the leaf indices for each training point
        Establish a dictionary for each unique leaf in DecisionTreeRegressor
        Go through each training example and append the index along with the current pseudo residual to list value associated with
        each unique leaf key.

        Use this dict to calculate the numerator for residual transformation into leaf output
        Use prev calculated probability for each residual that maps to the leaf to calculate the denominator for the leaf output
        """



        leaf_indices = self.model.apply(train_X)

        #below gets the unique leaves we have
        unique_leaves = np.unique(leaf_indices)
        leaf_output_dict = {}
        leaf_output_list = []


        for i in range(0, len(unique_leaves)):
            if unique_leaves[i] not in leaf_output_dict:
                leaf = unique_leaves[i]
                leaf_output_dict[leaf] = []


        for j in range(0, len(leaf_indices)):
            curr_leaf = leaf_indices[j]
            curr_pseudo_resid = self.new_pseudo_resids[j]
            leaf_output_dict[curr_leaf].append((j, curr_pseudo_resid))


        #conditional for first vs second tree and beyond
        if not self.first_tree:
            self.calc_leaf_output(leaf_indices, leaf_output_dict, leaf_output_list, unique_leaves)

        else:
            for k in range(0, len(unique_leaves)):
                curr_leaf = unique_leaves[k]

                curr_numerator = sum(j for i, j in leaf_output_dict[curr_leaf])
                curr_denominator = (self.first_pred * (1 - self.first_pred))*len(leaf_output_dict[curr_leaf])
                output = curr_numerator/curr_denominator

                leaf_output_list.append((curr_leaf, output))

        print("leaf output list: " + str(leaf_output_list))

        return leaf_output_list




    def predict_prob(self, X_train_numpy, y_train_numpy, leaf_output_list):


        """
        @X_train_numpy: Numpy array of shape (N, d,) consisting of the training features
        @y_train_numpy: Numpy array of shape (N,) consisting of the training target labels
        @leaf_output_list: tuple list of the new leaves and their outputs and calculated from the new pseudo residual calculation.

        Get the leaf outputs of the current training points.
        Then, goes through the tuple list to get the matching leaf and its output value from the tree.
        Uses it to make a new prediction.
        Converts the prediction into a probability using logistic regression function.
        Adds the prediction to a new probability list for that particular training point for use in the next boosting round.
        """


        log_odds_pred = 0
        actual_prob = 0
        count = 0
        print()
        #the i value is correct
        for i in range(0, len(X_train_numpy)):

            if count == 0:
                self.i = i
                count += 1


            reshaped_row = X_train_numpy[i].reshape(1, -1)
            curr_train_leaf = self.model.apply(reshaped_row)

            for item in leaf_output_list:

                if item[0] == curr_train_leaf[0]:
                    output = item[1]

                    #only use this calc with y pred if self.first_tree = true
                    if self.first_tree:
                        log_odds_pred = self.first_pred + (self.learning_rate*output)

                    else:
                        prev_prob = self.new_probs[i]
                        log_odds_pred = prev_prob + (self.learning_rate)*output

                    #convert log odds back into probability
                    actual_prob = (math.exp(log_odds_pred)/(1 + math.exp(log_odds_pred)))

                    if i == self.i:
                        print("the new probability for leaf: " + str(curr_train_leaf[0]) + ": " + str(actual_prob))


            #update the probabilities list if we're on the 2nd tree
            if self.first_tree:
                self.new_probs.append(actual_prob)
            else:
                self.new_probs[i] = actual_prob


        if self.first_tree:
            self.first_tree = False









    #@abstractmethod
    def predict(self, test_X: np.array) -> np.array:
        """
        Predict using the learnt classifier
        @test_X: Numpy array of shape (N, d,) consisting of the test features
        returns: Numpy array of shape (N,) consisting of the predicted labels
        """

        #go through test set and predict class output for x_train values
        y_test_pred = self.model.predict(test_X)

        return y_test_pred



    def accuracy(self, y_test_pred, test_y):
        """
        @y_test_pred: List of shape (N, d,) containing the test predictions
        @test_y: Numpy array of shape (N, d,) consisting of the test features

        Calculates accuracy on test set. Round up from 0.5 to verify prediction of 1, round down for 0
        """
        
        total_preds = len(test_y)
        correct_preds = 0
        curr_acc = 0


        for i in range(0, len(y_test_pred)):

            if y_test_pred[i] < 0.5 and test_y[i] == 0:
                correct_preds += 1

            else:
                if y_test_pred[i] >= 0.5 and test_y[i] == 1:
                    correct_preds += 1

        curr_acc = (correct_preds/total_preds)*100
        print("the current accuracy of this tree on test set: " + str(curr_acc))

        print("feature importances: " + str(self.model.feature_importances_))
        return curr_acc
