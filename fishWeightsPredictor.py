import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna
import warnings

#Warnings Suppressed for better user interface.
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

#Loading dataset.
filePath = input("Enter the file path of the fish dataset (CSV): ")
data = pd.read_csv(filePath)

#Encode the Species column for training purposes, and store the original Species column for checking purposes.
labelEncoder = LabelEncoder()
data["SpeciesEncoded"] = labelEncoder.fit_transform(data["Species"])  

#Define the Features(X) and Target(Y).
species_names = data["Species"].values 
X = data.drop(columns=["Weight", "Species"]) 
#Reshape the Weights column for use.
Y = data["Weight"].values.reshape(-1, 1)

#Scale features and target.
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y = scaler_Y.fit_transform(Y).ravel()  # Keep Y as a 1D array after scaling

#Split dataset int trainand test sets.
Xtrain, Xtest, Ytrain, Ytest, species_train, species_test = train_test_split(
    X, Y, species_names, test_size=0.2, random_state=17
)

#The Optuna objective function for hyperparameter optimization.
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.1),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "seed": 17,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(Xtrain, Ytrain, verbose=False)

    #Making predictions.
    predY = model.predict(Xtest)

    #Calculate evaluation metrics.
    rmse = np.sqrt(mean_squared_error(Ytest, predY))
    r2 = r2_score(Ytest, predY)

    #Maximize R^2 while minimizing RMSE.
    return r2 - rmse  

#Run the Optuna optimization process.
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

#Train final model with best parameters.
bestParameters = study.best_params
bestParameters.update({"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 17})

bestModel = xgb.XGBRegressor(**bestParameters)
bestModel.fit(Xtrain, Ytrain, verbose=False)

#Evaluating the model.
predY = bestModel.predict(Xtest)

#Reshape, and inverse transform the columns.
predY = scaler_Y.inverse_transform(predY.reshape(-1, 1)).ravel() 
Ytest = scaler_Y.inverse_transform(Ytest.reshape(-1, 1)).ravel()

#Compute RMSE and R^2.
rmse = np.sqrt(mean_squared_error(Ytest, predY))
r2 = r2_score(Ytest, predY)

#Print the evaluation results.
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

#Print 5 random actual vs predicted values along with species, for visulisation.
np.random.seed(17)
random_indices = np.random.choice(len(Xtest), size=5, replace=False)

print("\nSample Predictions:")
for i in random_indices:
    actual_weight = Ytest[i]
    predicted_weight = predY[i]
    species = species_test[i]  # Get species name
    print(f"Species: {species}; Actual Weight = {actual_weight:.2f}; Predicted Weight = {predicted_weight:.2f}")
