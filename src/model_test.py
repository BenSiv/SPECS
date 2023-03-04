"""
Testing models for the SPECS dataset
"""

# packages
# import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def read_data(datafile):
    """reads and prep data"""
    df = pd.read_csv(datafile)
    df.set_index(["consensus", "Plasmid"], inplace=True)
    features = df.drop(columns=["log2_median_fluorescence"])
    target = df["log2_median_fluorescence"]
    return target, features

def main(project_path, model):
    """runs model and returns stats"""
    train_file = os.path.join(project_path, "data/train_dat.csv")
    y_train, X_train = read_data(train_file)

    le = LabelEncoder()
    X_train["cell_line"] = le.fit_transform(X_train["cell_line"])

    test_file = os.path.join(project_path, "data/val_dat.csv")
    y_test, X_test = read_data(test_file)
    X_test["cell_line"] = le.transform(X_test["cell_line"])

    reg = model()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    return r2_score(y_test, y_pred)


if __name__ == "__main__":
    PROJECT_PATH = "/home/bensiv/Documents/ITC/Main/FinalProject/SPECS_Project/SPECS/"
    models_to_test = [LinearRegression, SVR, GradientBoostingRegressor, RandomForestRegressor]
    for model in models_to_test:
        print(f"{model.__name__}: {main(PROJECT_PATH, model):.0%}")
