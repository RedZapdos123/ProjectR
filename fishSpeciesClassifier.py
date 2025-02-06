#A python program to classify fishes from a fish's weight and dimensions, using XGBoost with Optuna (Baynesian optimizer) algorithm.
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings

#Suppressing all of the displayed warnings and optuna's determination of the hyperparameters' outputs.
#This is done to make the output on the terminal, better.
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)


#Taking the dataset file path as input, and laoding and converting the dataset into a dataframe.
filePath = input("Enter the file path of the fish species dataset (CSV): ")
data = pd.read_csv(filePath)

#Encoding the Species column of dataset into numbers.
labelEncoder = LabelEncoder()
data["Species"] = labelEncoder.fit_transform(data["Species"])

#Splitting up the target (Species) and features' columns.
X = data.drop("Species", axis=1)
#Converted the Species column into a 1D Numpy array to remove the warning.
Y = data["Species"].values.ravel()

#Splitiiing the dataset into training (80%) and testing (20%) sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 17)

#The Optuna's objective function.
def objective(trial):
    params ={
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "num_class": len(labelEncoder.classes_),
        "seed": 17,
    }

    #Training teh model.
    model = xgb.XGBClassifier(**params)
    model.fit(Xtrain, Ytrain)

    #Making prediction upon the test set.
    predY = model.predict(Xtest)

    # Calculate evaluation metrics
    f1 = f1_score(Ytest, predY, average = "weighted")
    accuracy = accuracy_score(Ytest, predY)
    precision = precision_score(Ytest, predY, average = "weighted", zero_division = 0)

    #Making attempt optuna to maximize the F1 and accuracy scores.
    return f1 + accuracy

#Making Optuna study (determine) the best hyperparameters and features.
study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 100)

#Training the model with best hyperparameters,a nd most required features.
bestParameters = study.best_params
bestParameters["objective"] = "multi:softprob"
bestParameters["eval_metric"] = "mlogloss"
bestParameters["num_class"] = len(labelEncoder.classes_)
bestParameters["seed"] = 17

bestModel = xgb.XGBClassifier(**bestParameters)
bestModel.fit(Xtrain, Ytrain)

#Evaluating the model.
predY = bestModel.predict(Xtest)

f1 = f1_score(Ytest, predY, average = "weighted")
accuracy = accuracy_score(Ytest, predY)
precision = precision_score(Ytest, predY, average = "weighted", zero_division = 0)
matrix = confusion_matrix(Ytest, predY)

#printing the evaluation metrics.
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print("Confusion Matrix:")
print(matrix)

#Printing five random classifications.
np.random.seed(17)
ra = np.random.choice(Xtest.index, size = 5, replace = False)
for i in ra:
    actual = labelEncoder.inverse_transform([Ytest[Xtest.index == i]])[0]
    predicted = labelEncoder.inverse_transform([bestModel.predict(Xtest.loc[i:i].values)])[0]
    print(f"Actual Species = {actual}; Predicted Species = {predicted}")