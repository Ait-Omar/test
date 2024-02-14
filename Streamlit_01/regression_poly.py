import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

#Chargement des données
data = pd.read_excel("Base AP.xls")
#Séparation des données en ensembles d'entraînement et de test
X = data[['x1', 'x2', 'x3']].values

y = data['Y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)
# Mise à l'échelle des données
Scaler=StandardScaler()
X_train_scaled= Scaler.fit_transform(X_train)
X_test_scaled= Scaler.fit_transform(X_test)

for degree in np.arange(1,11):
    polynomial_model= PolynomialFeatures(degree=degree)
    linear_model=LinearRegression()
    x_poly_train = polynomial_model.fit_transform(X_train)
    linear_model.fit(x_poly_train,y_train)
    def prediction(df):
        poly=polynomial_model.fit_transform(df)
        pred=linear_model.predict(poly)
        return pred
    
    model_filename = f'rg_poly_degree_{degree}.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(linear_model, model_file)

      