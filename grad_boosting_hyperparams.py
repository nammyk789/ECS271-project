from data import *
import optuna
from grad_boosting_interface import *



def convert_data(y):
    zero_count = 0
    one_count = 0

    for i in range(0, len(y)):
        if y[i] == 0:
            y[i] = 1
            one_count += 1
        elif y[i] == 1:
            y[i] = 0
            zero_count += 1
        else:
            y[i] = 0
            zero_count += 1

    return zero_count, one_count


train_X, train_y, val_X, val_y, test_X, test_y = get_data()

#making two class problem for gradient boosting due to simplicity
#for grad boosting, convert classes for train_y, test_y, and val_y to 0 and 1
train_zero_count, train_one_count = convert_data(train_y)
val_zero_count, val_one_count = convert_data(val_y)
test_zero_count, test_one_count = convert_data(test_y)


#need to just add max features
def objective(trial):

    max_depth = trial.suggest_int("max_depth", 3, 4)
    #max_features = trial.suggest_float("max_features", 2, 21),
    leaves = trial.suggest_int("leaves", 3, 8)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1)
    print("the learning rate: " + str(learning_rate))
    boosting_rounds = trial.suggest_int("boosting_rounds", 1, 5)

    #create decision tree regressor
    model = DecisionTreeRegressor(criterion='absolute_error', max_depth=max_depth, max_leaf_nodes=leaves)

    #pass into the interface
    grad_boosting = Grad_Boosting_Interface(train_zero_count, train_one_count, model, learning_rate, boosting_rounds)
    grad_boosting.train(train_X, train_y, val_X, val_y)
    y_test_pred = grad_boosting.predict(test_X)
    return grad_boosting.accuracy(y_test_pred, test_y)




study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)
