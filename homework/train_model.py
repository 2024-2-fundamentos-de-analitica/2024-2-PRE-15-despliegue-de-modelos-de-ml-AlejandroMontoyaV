"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import os
os.chdir("C:/Users/Alejandro Montoya V/Desktop/2024-2-PRE-15-despliegue-de-modelos-de-ml-AlejandroMontoyaV/homework")

df = pd.read_csv("house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)

with open("house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)
    