import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import optuna
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS

#Suppressing the warnings for cleaner output, on the terminal.
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = DataConversionWarning)

#Taking the dataset file path as input, and laoding it up.
filePath = "C:\\Users\\Xeron\\OneDrive\\Documents\\Programs\\MachineLearning\\ProjectRv1\\FishSpeciesData.csv"
data = pd.read_csv(filePath)

#Encoding the Species column of the dataset.
labelEncoder = LabelEncoder()
data["Species"] = labelEncoder.fit_transform(data["Species"])

#Splitting the features and target (encoded Species) columns
X = data.drop("Species", axis = 1)
Y = data["Species"].values.ravel()

#Splitting the dataset into training (80%) and testing (20%) sets. 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=17)

#The Optuna objective function, to be used for the determination of the best hyperparameters.
def objective(trial):
    params = {
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

    model = xgb.XGBClassifier(**params)
    model.fit(Xtrain, Ytrain)
    predY = model.predict(Xtest)

    return f1_score(Ytest, predY, average = "weighted") + accuracy_score(Ytest, predY)

#Creating a 'study' to find the best hyperparameters.
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

#Training the model with best hyperparameters, as found by Optuna, using the baynesian optimizer.
bestParameters = study.best_params
bestParameters.update({
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": len(labelEncoder.classes_),
    "seed": 17
})

bestModel = xgb.XGBClassifier(**bestParameters)
bestModel.fit(Xtrain, Ytrain)

#Creating the flask 'app', for linking up the model to the website.
app = Flask(__name__)
#Using Flask CORS for cross origin requests, to solve the port difference problem.
CORS(app)

@app.route('/predict', methods = ['POST'])

#Doing a predicted classification with user inputed values.
def predict():
    try:
        inputData = request.get_json()
        features = np.array([inputData[key] for key in X.columns]).reshape(1, -1)

        #Inversing the transformation with label encoder's function.
        predictedSpecies = labelEncoder.inverse_transform(bestModel.predict(features))[0]

        #Evaluating the model and displaying it on the website. The Evaluation metrics are: F1 score, accuracy  precision.
        predY = bestModel.predict(Xtest)
        metrics ={
            "F1 Score": round(f1_score(Ytest, predY, average = "weighted"), 4),
            "Accuracy": round(accuracy_score(Ytest, predY), 4),
            "Precision": round(precision_score(Ytest, predY, average = "weighted", zero_division = 0), 4)
        }

        return jsonify({"Predicted Species": predictedSpecies, "Metrics": metrics})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug = True)
