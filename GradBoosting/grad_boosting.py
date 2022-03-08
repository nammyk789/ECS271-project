import typing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import tree
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
from statistics import mean
import math
import warnings





class Grad_Boosting:

    def __init__(self, one_count, zero_count):
        self.one_count = one_count
        self.zero_count = zero_count
        self.first_tree = True
        #create new dict here for leaf node index: predicted probability in current boosting round
        self.new_probs = []
        self.first_pred = 0
        self.new_pseudo_resids = []
        self.i = 0
        self.loss_list = []
        self.curr_pred = 0
        self.denominator = 0

    '''
    Takes in a model and performs gradient boosting using that model. This allows for almost any scikit-learn
    model to be used.
    '''
    def GradBoostClassifier(self, model,
                  X_test,
                  X_train,
                  y_train,
                  learning_rate,
                  boosting_rounds,
                  ) -> np.array:

        X_train_numpy = X_train.to_numpy()
        y_train_numpy = y_train.to_numpy()
        #instead of getting the mean of the y training data, instead we calc log(odds) of normal fetus and then
        #pass into logistic function
        log_odds = math.log(self.one_count/self.zero_count)
        print("log odds: " + str(log_odds))

        #now, plug into logistic regression equation
        log_regression = math.exp(log_odds)/(1 + math.exp(log_odds))

        self.first_pred = log_regression
        print("prediction for first boosting round: " + str(self.first_pred))
        #now we need to calculate residuals (pseudo residuals) for each class - y_train - in our data

        self.new_pseudo_resids = [0]*len(y_train_numpy)
        count = 0

        for elem in range(0, len(y_train_numpy)):
            self.new_pseudo_resids[count] = y_train_numpy[elem] - self.first_pred
            count += 1

        #saved all the pseudo resids
        #print("pseudo resids: " + str(self.new_pseudo_resids[0:10]))


        #fit the decisiontree regressor to the x_train data and pseudo residuals
        # we have 21 features in this data
        model = model.fit(X_train_numpy, self.new_pseudo_resids)
        tree.plot_tree(model)
        plt.show()

        leaf_output_list = self.apply_residual_transformation(model, X_train_numpy)

        #initialize new list to store probabilities and predict the new probability
        self.predict_prob(model, X_train_numpy, y_train_numpy, leaf_output_list, learning_rate)


        #get new pseudo residuals
        #y_train - pred_list

        self.new_pseudo_resids = [0]*len(y_train_numpy)


        for i in range(0, len(y_train_numpy)):

            self.new_pseudo_resids[i] = y_train_numpy[i] - self.new_probs[i]


        self.loss_list.append(sum(self.new_pseudo_resids))






        #for loop through boosting rounds
        #fit model to new pseudo residuals
        #apply the residual transformation on the model (apply_residual_transformation)
        #predict the probability of each leaf from the outputs for each training point (predict_prob)
        #update the new pseudo residuals (like above)
        curr_acc = []
        total_preds = len(y_train_numpy)

        #repeat
        for j in range(0,boosting_rounds):

            print()
            print("boosting round: " + str(2 + j))

            model = model.fit(X_train_numpy, self.new_pseudo_resids)
            leaf_output_list = self.apply_residual_transformation(model, X_train_numpy)
            self.predict_prob(model, X_train_numpy, y_train_numpy, leaf_output_list, learning_rate)

            correct_preds = 0
            for k in range(0, len(y_train_numpy)):
                if self.new_probs[k] >= 0.5 and y_train_numpy[k] == 1:
                    correct_preds += 1

                else:
                    if self.new_probs[k] < 0.5 and y_train_numpy[k] == 0:
                        correct_preds += 1

                self.new_pseudo_resids[k] = y_train_numpy[k] - self.new_probs[k]


            curr_acc = (correct_preds/total_preds)*100
            print("the current training accuracy for epoch " + str(j) + ": " + str(curr_acc))
            print()
            self.loss_list.append(sum(self.new_pseudo_resids))

            #calculate accuracy for this round (epoch)

            #print("the new pseudo_resids: " + str(self.new_pseudo_resids[0:10]))
            #print()

        #plotting the final tree
        tree.plot_tree(model)
        plt.show()
        print("the loss list: " + str(self.loss_list))
        print()


        return model


    """get the leaf outputs of the current training points. then, go through the tuple list to get the matching leaf and its output
    value from the tree. then, use it to make a new prediction. probably should initialize the new probability dict here first for all the leaves."""
    def predict_prob(self, model, X_train_numpy, y_train_numpy, leaf_output_list, learning_rate):



        #need to loop through training data, do model.apply to get leaf
        #return the output value from the tuple list stored
        log_odds_pred = 0
        actual_prob = 0
        count = 0
        print()
        #the i value is correct
        for i in range(0, len(X_train_numpy)):

            if count == 0:
                curr_val = X_train_numpy[i]
                print("the current training point: " + str(curr_val))
                print(curr_val[8])
                print(curr_val[17])
                print("expected pred for this training point: " + str(y_train_numpy[i]))
                self.i = i
                count += 1


            reshaped_row = X_train_numpy[i].reshape(1, -1)
            #note, may need to change this
            curr_train_leaf = model.apply(reshaped_row)

            if self.i == i:
                print("the current training leaf: " + str(curr_train_leaf))
                print()

            #use the current leaf we've gone to in order to determine the output for log(odds)
            for item in leaf_output_list:

                if item[0] == curr_train_leaf[0]:
                    output = item[1]
                    #only use this calc with y pred if self.first_tree = true
                    if self.first_tree:
                        log_odds_pred = self.first_pred + (learning_rate*output)

                        if i == self.i:
                            print("first pred: " + str(self.first_pred))
                            print("lr: " + str(learning_rate))
                            print("output: " + str(output))
                            print("log odds pred: " + str(log_odds_pred))
                            print()

                    else:
                        prev_prob = self.new_probs[i]
                        log_odds_pred = prev_prob + (learning_rate)*output

                        if i == self.i:
                            print("curr pred: " + str(prev_prob))
                            print("lr: " + str(learning_rate))
                            print("output: " + str(output))
                            print("log odds pred: " + str(log_odds_pred))
                            print()


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




    #this function is used in particular for the second tree and beyond
    #we need to take in the leaf output dict as input as well as leaf indices
    #in order to calculate the output for a leaf, we need to go through all the residuals for that leaf and subtract from their previous probability pred

    #this will be done by going through all the residuals in leaf_output_dict[curr_leaf] and accessing their previous probability (self.new_probs[i])
    #then subtract these from 1
    def calc_leaf_output(self, leaf_indices, leaf_output_dict, leaf_output_list, unique_leaves):

        curr_numerator = 0
        curr_denominator = 0

        prev_prob = 0

        for i in range(0, len(unique_leaves)):
            curr_leaf = unique_leaves[i]
            print("the current leaf: " + str(curr_leaf))
            #numerator is calc in same way- the sum of the residuals
            curr_numerator = sum(j for i, j in leaf_output_dict[curr_leaf])



            #for the denom, we get the index of the current training point from the leaf_output_dict tuple
            #then, we get the current prob for that training point by checking new_prob_list

            curr_leaf_indices = [x for x in leaf_indices if x == curr_leaf]

            if curr_leaf == 4:
                print("the current numerator: " + str(curr_numerator))
                print("length: " + str(len(curr_leaf_indices)))


            for j in range(0, len(curr_leaf_indices)):
                train_leaf = curr_leaf_indices[j]
                index = leaf_output_dict[train_leaf][j][0]

                prev_prob = self.new_probs[index]
                curr_denominator += prev_prob * (1 - prev_prob)

            print("the current denominator: " + str(curr_denominator))

            output = curr_numerator/curr_denominator
            print("the output: " + str(output))
            leaf_output_list.append((curr_leaf, output))
            print()


        return leaf_output_list


    #note: can use model.apply to see where leaf maps to
    #apply the transformation of the residual since we're doing a classification
    def apply_residual_transformation(self, model, X_train_numpy):
        leaf_indices = model.apply(X_train_numpy)
        #below gets the unique leaves we have
        unique_leaves = np.unique(leaf_indices)
        print(unique_leaves)
        leaf_output_dict = {}
        leaf_output_list = []


        for i in range(0, len(unique_leaves)):
            if unique_leaves[i] not in leaf_output_dict:
                leaf = unique_leaves[i]
                leaf_output_dict[leaf] = []

        print(leaf_output_dict)

        #go through each training example (len(leaf_indices))
        #if we're creating the first tree, then append (index, pseudo_residual[i]) to leaf_index[i] inside the leaf output dict

        for j in range(0, len(leaf_indices)):
            curr_leaf = leaf_indices[j]
            curr_pseudo_resid = self.new_pseudo_resids[j]
            leaf_output_dict[curr_leaf].append((j, curr_pseudo_resid))


        #function to calc for second tree
        if not self.first_tree:
            self.calc_leaf_output(leaf_indices, leaf_output_dict, leaf_output_list, unique_leaves)

        else:
            #now we calc the numerator
            for k in range(0, len(unique_leaves)):
                curr_leaf = unique_leaves[k]

                curr_numerator = sum(j for i, j in leaf_output_dict[curr_leaf])
                #print("the numerator: " + str(curr_numerator))

                #add up all the resids for the prev prob as denom. in the first tree, these are all the same
                curr_denominator = (self.first_pred * (1 - self.first_pred))*len(leaf_output_dict[curr_leaf])
                #print("the denominator: " + str(curr_denominator))
                output = curr_numerator/curr_denominator

                print("the output: " + str(output))
                leaf_output_list.append((curr_leaf, output))
                print()

        print("leaf output list: " + str(leaf_output_list))

        return leaf_output_list










    #function to run model on test set
    def test_model(self, tree_model, x_test, y_test):

        x_test_numpy = x_test.to_numpy()
        y_test_numpy = y_test.to_numpy()
        #go through test set and predict class output for x_train values
        total_preds = len(y_test)
        curr_acc = 0

        correct_preds = 0
        print("x test numpy shape: " + str(x_test_numpy[0].shape))

        y_test_pred = tree_model.predict(x_test_numpy)

        #round up from 0.5 to verify prediction of 1, round down for 0
        for i in range(0, len(y_test_pred)):

            if y_test_pred[i] < 0.5 and y_test_numpy[i] == 0:
                correct_preds += 1

            else:
                if y_test_pred[i] >= 0.5 and y_test_numpy[i] == 1:
                    correct_preds += 1

        curr_acc = (correct_preds/total_preds)*100
        print("the current accuracy of this tree on test set: " + str(curr_acc))

        print("feature importances: " + str(tree_model.feature_importances_))












def create_split_and_learner(x, y):

    n_round = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

    #need to remove header


    #decisiontreeregressor weak learner
    #change squared to 0-1 error
    #change number of leaves to between 8 and 32 later
    tree_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=3, max_leaf_nodes=8)
    return x_train, x_test, y_train, y_test, tree_model



def main():

    df = pd.read_csv('fetal_health.csv')
    one_count = 0

    fetal_health_col = df['fetal_health']
    zero_count = 0
    one_count = 0
    n_round = 0

    for i in range(0, len(fetal_health_col)):
        if fetal_health_col[i] == 1:
            #trying to make this two class problem for simplicity first
            one_count += 1
        elif fetal_health_col[i] == 2:
            fetal_health_col[i] = 0
            zero_count += 1
        else:
            fetal_health_col[i] = 0
            zero_count += 1

    #print("one count: " + str(one_count))
    #print()
    #print("zero count: " + str(zero_count))
    #print()
    #print(len(fetal_health_col))

    #1655 - cat 1
    #print(one_count)

    #295- cat 2
    #print(two_count)

    #176- cat 3
    #print(three_count)

    #log odds of each event?

    #each point in x_train will be array of features in fetal health dataset

    #we only need the other cols besides fetal health for dataset
    x = df.drop(["fetal_health"], axis = 1)
    y = fetal_health_col

    x_train, x_test, y_train, y_test, tree_model = create_split_and_learner(x, y)
    boost_class = Grad_Boosting(one_count, zero_count)
    boosting_rounds = 1
    learning_rate = 0.01
    tree_model = boost_class.GradBoostClassifier(tree_model, x_test, x_train, y_train, learning_rate, boosting_rounds)

    #run on test set data now
    boost_class.test_model(tree_model, x_test, y_test)
    print()
    print()




main()
