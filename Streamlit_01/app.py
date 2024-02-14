import streamlit as st
import time
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import ann

#import matplotlib.pyplot as plt
#import plotly.express as px
#import plotly.graph_objects as go

# Définir la couleur de fond de l'application
st.set_page_config(layout="wide", page_title="ViscoPredictor Pro", page_icon=":chart_with_upwards_trend:", initial_sidebar_state="expanded",)
# Styles CSS pour le fond de l'application

#logo
#logo = Image.open("logo.png")
#logo_resized = logo.resize((100, 50))
#st.sidebar.image(logo_resized, caption='digitalization, innovation & process simulation', use_column_width=True)
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;  /* Changer la couleur de fond ici */
            font-size: 16px;
            font-family: 'Arial', sans-serif;
            margin: 0;
            background-image: url("background.jpg");
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .styled-table {
            background-color: white;
            padding: 10px;
            border-radius: 0px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            margin: auto;
            text-align: center;
            text-color: white;
        }
            .styled1-table {
            background-color: white;
            border-collapse: transparent;
            color: wihte;
            font-weight: bold;
            padding: 10px;
            border-radius: 0px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            margin: auto;
            text-align: center;
        }
        .styled-box {
            background-color: #3498db;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .custom-button {
            border-color: transparent;
        }
    </style>
""", unsafe_allow_html=True)
# Titre principal
def traitement_long():
    time.sleep(5)
title="Réacteur d'attaque de la prduction d'acide phosphorique"
st.markdown(f"<h1 style='color: blue; text-align: center;background-color:powderblue;'>{title}</h1>", unsafe_allow_html=True)
# Introduction
st.markdown("<h2 style='text-align: center;'>Prévision de la viscosité dynamique de la bouillie utilisant machine learning</h2>", unsafe_allow_html=True)

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

st.sidebar.header("Paramètres du fonctionement")
# Sidebar pour les paramètres
def user_input():
    tx1 = "σ: Shear stress (100 à 980 )"
    tx2 = "T: Temperature (75°C à 85°C )"
    tx3 = "SC: Solid content (31% à 35%)"
    SR = float(st.sidebar.text_input(tx1,821))
    T = float(st.sidebar.text_input(tx2,75.95))
    SC = float(st.sidebar.text_input(tx3,31.46))
    data={'SR':SR,
          'T':T,
          'SC':SC,
          }
    parametres=pd.DataFrame(data,index=["Paramètres"])
    return parametres
df=user_input()
param_de_fonctionnement=df
# Affichage des paramètres dans un tableau
donnees_tableau = [
  ['σ', 'T', 'SC'],
  [float(param_de_fonctionnement['SR'].values), float(param_de_fonctionnement['T'].values),float(param_de_fonctionnement['SC'].values)]
                ]
df_tableau = pd.DataFrame(donnees_tableau)
# Convertir le DataFrame en HTML en masquant les index et en-têtes
html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
# Centrer le tableau avec du CSS
style = """
    <style>
        .styled-table {
            margin: auto;
            text-align: center;
        }
    </style>
"""
# Afficher le tableau avec st.markdown
if st.sidebar.checkbox('afficher les paramètres sélectionnées '):
  st.markdown("<p style='text-align: center;'>Paramètres du fonctionnement sélectionnés:</p>", unsafe_allow_html=True)
  st.markdown(style + html_tableau, unsafe_allow_html=True)
st.write("  ")
# Modèles ANN
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
# difinition des fonctions de Prédiction
def prediction1(df):
  pred=model1.predict(df)
  return pred
def prediction2(df):
  pred=model2.predict(df)
  return pred
#la metrique d'evaluation
def metric_ecalu():
  st.write(" ")
  list_metrics=["Options","RMSE","MAE"]
  metric = st.sidebar.selectbox("choisir une métrique d'évaluation",list_metrics)
  if st.sidebar.checkbox("afficher"):
    if (classifier == "ANN" and (optimizer == "Adam" and metric == "RMSE")):
        donnees_tableau = [
            ['RMSE'],
            [rmse1]
                    ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    elif (classifier == "ANN" and (optimizer == "Adam" and metric == "MAE")):
        donnees_tableau = [
            ['MAE'],
            [mae1]
                    ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    elif (classifier == "ANN" and (optimizer == "SGD" and metric == "RMSE")):
        donnees_tableau = [
          ['RMSE'],
            [rmse2]
                  ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    elif (classifier == "ANN" and(optimizer == "SGD" and metric == "MAE")):
        donnees_tableau = [
            ['MAE'],
            [mae2]
                  ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    elif (classifier == "Régression polynomial" and metric == "RMSE"):
        donnees_tableau = [
          ['RMSE'],
          [rmse]
                  ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    elif (classifier == "Régression polynomial" and metric == "MAE"):
        donnees_tableau = [
          ['MAE'],
          [mae]
                  ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
        st.markdown(html_tableau, unsafe_allow_html=True)
    else:
     st.write(" ")
#choisir le modèle ML
list_classifierr=["Options","Régression polynomial", "ANN"]
st.sidebar.subheader("Chosir un modèl")
classifier = st.sidebar.selectbox("modèls",list_classifierr)
#selection des paramètres des modèles
if classifier == "ANN":
  st.sidebar.header("Hyperparamètres du modèle")
  list_optimizer=['','Adam','SGD']
  optimizer=st.sidebar.selectbox("Chosir l'optimisateur",list_optimizer)
  nombre_neurones=st.sidebar.number_input('Initier le nombre des neurones (1 à 20)',1,21,14)
  list_fct_activation=["relu","leaky_relu","tanh","linear","sigmoid"]
  fonction_activation=st.sidebar.selectbox("Choisir la fonction d'activation",list_fct_activation)

  donnees_tableau = [
  ['Optimisateur', 'Nombre des neurones', 'Fonction d\'activation'],
  [optimizer, nombre_neurones,fonction_activation]
                ]
  df_tableau = pd.DataFrame(donnees_tableau)
# Convertir le DataFrame en HTML en masquant les index et en-têtes
  html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
# Centrer le tableau avec du CSS
  style = """
    <style>
        .styled-table {
            margin: auto;
            text-align: center;
        }
    </style>
  """
# Afficher le tableau avec st.markdown
  if st.sidebar.checkbox('afficher les paramètres sélectionnées'):
    st.markdown("<p style='text-align: center;'>Paramètres du modèle sélectionnées:</p>", unsafe_allow_html=True)
    st.markdown(style + html_tableau, unsafe_allow_html=True)
# Affichage des résultats
  st.write(" ")
  st.markdown("<p style='color: #808080;text-align: center;'>Cliquer sur le boutton Execution pour afficher  les résultats.</p>", unsafe_allow_html=True)
  message_container = st.empty()
  if optimizer=="Adam":
      if st.sidebar.button("Execution"):
        #creation des modèles
        #st.text("Attend, ça prend quelques secondes...")
        #st.markdown("<p style='text-align: center;'>Traitement en cours...</p>", unsafe_allow_html=True)
        message_container.markdown(
            "<p style='text-align: center;color: red;'>Traitement en cours...</p>",
                   unsafe_allow_html=True)
        #message_container.text("Traitement en cours...")
        
        model1 = ann.create_model1(nombre_neurones,fonction_activation)
        #model1.fit(X_train, y_train, epochs=100, batch_size=10,verbose=0,validation_data=(X_test,y_test))
        traitement_long()
        message_container.markdown(
             "<p style='text-align: center;color: green;'>Traitement terminé!</p>",
                unsafe_allow_html=True)
        #Metiques d'évaluation
        y_pred1 = model1.predict(X_test)
        rmse1 = np.sqrt(mean_squared_error(y_test, y_pred1))
        mae1 = mean_absolute_error(y_test, y_pred1)
        donnees_tableau = [ ['Viscosité prédite',"RMSE","MAE"],
                          [round(float(prediction1(df)),2), round(rmse1,2), round(mae1,2)]
                        ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled1-table"])
        st.markdown("<p style='text-align: center;'>Résultats obtenues:</p>", unsafe_allow_html=True)
        st.markdown(html_tableau, unsafe_allow_html=True)
      
  elif optimizer=="SGD":
      if st.sidebar.button("Execution"):
        message_container.markdown(
            "<p style='text-align: center;color: red;'>Traitement en cours...</p>",
                   unsafe_allow_html=True)
        #creation des modèles
        model2 = ann.create_model2(nombre_neurones,fonction_activation)
        #model2.fit(X_train, y_train, epochs=100, batch_size=10,verbose=0,validation_data=(X_test,y_test))
        traitement_long()
        message_container.markdown(
             "<p style='text-align: center;color: green;'>Traitement terminé!</p>",
                unsafe_allow_html=True)
        #Metiques d'évaluation
        y_pred2 = model2.predict(X_test)
        rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
        mae2 = mean_absolute_error(y_test, y_pred2)
        donnees_tableau = [['Viscosité prédite', "RMSE", "MAE"],
                        [round(float(prediction2(df)),2), round(rmse2,2), round(mae2,2)]
                        ]
        df_tableau = pd.DataFrame(donnees_tableau)
        html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled1-table"])
        st.markdown("<p style='text-align: center;'>Résultats obtenues:</p>", unsafe_allow_html=True)
        st.markdown(style + html_tableau, unsafe_allow_html=True)
    
  else:
    if st.sidebar.button("Execution"):
      st.markdown("<p style='color: red; text-align: center;'>Vous devez sélectionner un optimizateur!</p>", unsafe_allow_html=True)
elif classifier == "Régression polynomial":
  st.sidebar.header("Hyperparamètres du modèle")
  degree = st.sidebar.number_input("dgree",0,20,5)
  # Modèle de régression polynomiale
  polynomial_model= PolynomialFeatures(degree=degree)
  linear_model=LinearRegression()
  x_poly_train = polynomial_model.fit_transform(X_train)
  linear_model.fit(x_poly_train,y_train)
  def prediction(df):
    poly=polynomial_model.fit_transform(df)
    pred=linear_model.predict(poly)
    return pred
  poly=polynomial_model.fit_transform(X_test)
  y_pred = linear_model.predict(poly)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  mae = mean_absolute_error(y_test, y_pred)
  coefficient_correlation, _ = pearsonr(y_test, y_pred)
 
  donnees_tableau = [ ['degree'],
                      [degree]
                    ]
  df_tableau = pd.DataFrame(donnees_tableau)
  # Convertir le DataFrame en HTML en masquant les index et en-têtes
  html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled-table"])
  # Centrer le tableau avec du CSS
  style = """
    <style>
        .styled-table {
            margin: auto;
            text-align: center;
        }
    </style>
  """
  # Afficher le tableau avec st.markdown
  if st.sidebar.checkbox('afficher les paramètres sélectionnées  '):
    st.markdown("<p style='text-align: center;'>Paramètres du modèle sélectionnées:</p>", unsafe_allow_html=True)
    st.markdown(style + html_tableau, unsafe_allow_html=True)
  st.markdown("<p style='color: #808080;text-align: center;'>Cliquer sur le boutton Executer pour afficher  les résultats.</p>", unsafe_allow_html=True)
  if st.sidebar.button("Execution"):
      donnees_tableau = [ ['Viscosité prédite',"RMSE", "MAE","coefficient de corrélation"],
                          [round(float(prediction(df)),2), round(rmse,2), round(mae,2),round(coefficient_correlation,2)]
                        ]
      df_tableau = pd.DataFrame(donnees_tableau)
      html_tableau = df_tableau.to_html(index=False, header=False, classes=["styled1-table"])
      st.markdown("<p style='text-align: center;'>Résultats obtenues:</p>", unsafe_allow_html=True)
      st.markdown(html_tableau, unsafe_allow_html=True)

      #fig = px.scatter(data,x=y_test,y=y_pred,labels={'x':"viscosité réelle", 'y':"viscosité prédite"})
      
      #st.plotly_chart(fig)
else :
  if st.sidebar.button("Execution"):
   st.markdown("<p style='color: red; text-align: center;'>Vous devez sélectionner un modèle!</p>", unsafe_allow_html=True)

if st.sidebar.button("Méthodologie"):
   st.markdown("<h2>1. Introduction</h2>", unsafe_allow_html=True)
   st.markdown('''<p>  Ce site présente les résultats obtenus à partir d’une base de données. Après une
     analyse initiale des données, un modèle de réseau de neurones a été développé pour
     prédire la relation entre les variables.</p>''', unsafe_allow_html=True)
   st.markdown("<h2>2. Analyse des données</h2>", unsafe_allow_html=True)
   st.markdown("<h3>2.1 Boîte à moustaches</h3>", unsafe_allow_html=True)
   # Construire la boîte à moustaches avec Plotly Express
   #fig = px.box(data,labels=['x1', 'x2', 'x3','y'],title='figure 1: Boîte à moustaches pour chaque paramètre')
   #fig.update_layout(title=dict(x=0.5, y=0.95, xanchor='center', yanchor='top'))
   #st.plotly_chart(fig)
   st.markdown('''<p>  La boîte à moustaches permet de visualiser la distribution, la médiane, et les éventuelles
     valeurs aberrantes de chaque paramètre. Elle donne un aperçu rapide de la répartition
     des données pour chaque variable.</p>''', unsafe_allow_html=True)
   st.markdown("<h3>2.2 Matrice de corrélation</h3>", unsafe_allow_html=True)
   corr_matrix = np.corrcoef(data, rowvar=False)
   sns.set()
   plt.figure(figsize=(10, 8))
   fig=sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,xticklabels=data.columns,yticklabels=data.columns)
   plt.title('figure 2: Matrice de Corrélation')
   st.pyplot(fig.figure)
   st.markdown('''<p>  La matrice de corrélation en Figure 2 détaille les coefficients de corrélation entre chaque
    paire de variables. Une analyse attentive révèle des coefficients de l’ordre de 10−3
    entre x1 et x2, ainsi qu’entre x1 et x3, et de l’ordre de 10−2 entre x2 et x3. Ces valeurs relativement
    faibles traduisent une faible corrélation entre ces variables explicatives, indiquant ainsi
    leur indépendance. Cette indépendance est favorable car elle suggère que chaque variable
    apporte une information unique au modèle. De plus, une corrélation notable, de l’ordre
    de 10−1, est observée entre la variable cible y et les variables explicatives xi
    , témoignant de l’influence significative de ces dernières sur 
    la variable dépendante.</p>''', unsafe_allow_html=True)


