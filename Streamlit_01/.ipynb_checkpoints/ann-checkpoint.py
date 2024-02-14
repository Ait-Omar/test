import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle

#Chargement des données
data = pd.read_excel("Base AP.xls")
#Séparation des données en ensembles d'entraînement et de test
X = data[['x1', 'x2', 'x3']].values

y = data['Y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=100)


def create_model1(nb_neurons,fct_activation):
    model = Sequential()
    model.add(Dense(nb_neurons, activation=fct_activation))
    model.add(Dense(1, activation='linear')) # Output layer
    model.compile(Adam(learning_rate=0.001), 'mean_squared_error')
    return model
def create_model2(nb_neurons,fct_activation):
    model = Sequential()
    model.add(Dense(nb_neurons, activation=fct_activation))
    model.add(Dense(1, activation='linear')) # Output layer
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

 
list_fct_activation=["relu","leaky_relu","tanh","linear","sigmoid"]
for fct in list_fct_activation:
    for nb_neurons in np.arange(1,21):

        mod_adam=create_model1()
        mod_adam.fit(X_train, y_train, epochs=100, batch_size=10,verbose=0,validation_data=(X_test,y_test))

        mod_sgd=create_model2()
        mod_sgd.fit(X_train, y_train, epochs=100, batch_size=10,verbose=0,validation_data=(X_test,y_test))
        
        model_filename = f'ANN_adam_{nb_neurons}_{fct}.pkl'
        with open(model_filename, 'wb') as model_file:
          pickle.dump(mod_adam, model_file)
        model_filename = f'ANN_sgd_{nb_neurons}_{fct}.pkl'
        with open(model_filename, 'wb') as model_file:
          pickle.dump(mod_sgd, model_file)
        