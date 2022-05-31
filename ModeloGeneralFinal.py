# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:58:50 2022

@author: Alberto Jose Gutierrez Megias
"""

# -- LIBRERIAS Y OPCIONES -- #

"""
Librerías utilizadas
"""

import os
import pandas as pd
from osgeo import gdal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

import xgboost as xgb
from xgboost import plot_importance
from xgboost import cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# -- FUNCIONES - #

"""
-- Inicio de las funciones de carga y tratamiento de datos --
"""

"""
Carga de datos

@param: path -> directorio donde se encuentran los datos a cargar de la zona a la que se refiere la función
@return: datos
"""
def cargar_datos_SantoTome(path):
    print("\n-- Cargando datos Santo Tome --")
    dir = path
    ficheros=[]
    nombre=[]
    
    # Recorre todos los archivos del directorio y recoge solo los archivos con terminación .tif
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            ficheros.append(os.path.join(dir, file))
            nombre.append(file.split('.')[0])

    # Ordenado los ficheros y los nombres de los mismos por orden alfabético
    sorted(ficheros)
    sorted(nombre)
    
    # Crea 3 dataframe con las columnas de los nombres de las variables
    # Datos se utilizará para todo el proceso de aprendizaje
    datos = pd.DataFrame(columns=nombre)
    
    # Abrimos uno de los archivos TIF con la librería GDAL para obtener las dimensiones de las variables
    tif = gdal.Open(ficheros[1])
    tif_band = tif.GetRasterBand(1)
    x = tif_band.XSize
    y = tif_band.YSize
    
    # Recorremos cada uno de los ficheros del directorio
    contador = 0
    for f in ficheros:
        n = nombre[contador]
        MDT = gdal.Open(f)

        # Patrones sin datos se incluyen como -1
        MDT.GetRasterBand(1).SetNoDataValue(-1)

        # Pasamos los datos a float
        band = MDT.GetRasterBand(1).ReadAsArray().astype(float)

        # Se desechan las dos primeras filas y columnas debido a calculos de borde
        band = band[1:y, 1:x]

        # Reconstruimos el array con el tamaño de filas x columnas
        x_recorte = x - 1
        y_recorte = y - 1
        band = np.reshape(band, x_recorte*y_recorte)

        # Estos serán nuestros datos
        datos[n] = band

        contador = contador + 1
    
    # Devolvemos los datos
    return datos

"""
Tratamiemto de los datos

@param: datos -> datos a tratar, en este caso deben de ser de Santo Tomé
@return: datos
"""
def tratamiento_datos_SantoTome(datos):
    print("\n-- Tratamiento datos Santo Tome --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    # Se eliminan todos los datos que coinciden con los valores mostrados a continuación
    # Ej. Solo queremos los datos donde la distancia de carreteras sea mayor o igual que 0 el resto serán datos corruptos
    datos = datos[datos['Carcavas'] != 128]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Distancia_Carreteras'] >= 0]
    datos = datos[datos['Geologia'] != 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
    print("Número de datos despues del tratamiento ", datos.size)   
    # Tratamiento de valores categóricos dependiendo del estándar establecido
    
    # Geología
    datos.loc[datos.Geologia == 1, "Geologia"] = "Codigo_9131"
    datos.loc[datos.Geologia == 2, "Geologia"] = "Codigo_9330"
    datos.loc[datos.Geologia == 3, "Geologia"] = "Codigo_9132"
    datos.loc[datos.Geologia == 4, "Geologia"] = "Codigo_9133"
    datos.loc[datos.Geologia == 5, "Geologia"] = "Codigo_9001"
    datos.loc[datos.Geologia == 6, "Geologia"] = "Codigo_9002" 
    
    # Usos del suelo
    datos.loc[datos.Usos_Del_Suelo == 1, "Usos_Del_Suelo"] = "Tejido_urbano"
    datos.loc[datos.Usos_Del_Suelo == 2, "Usos_Del_Suelo"] = "Labor_secano"
    datos.loc[datos.Usos_Del_Suelo == 3, "Usos_Del_Suelo"] = "Tierras_regadas"
    datos.loc[datos.Usos_Del_Suelo == 4, "Usos_Del_Suelo"] = "Olivares"
    datos.loc[datos.Usos_Del_Suelo == 5, "Usos_Del_Suelo"] = "Mosaicos_cultivos"
    datos.loc[datos.Usos_Del_Suelo == 6, "Usos_Del_Suelo"] = "Cursos_agua"
    
    # Unidades edáficas
    datos.loc[datos.Unidades_Edaficas == 44, "Unidades_Edaficas"] = "Codigo_44"
    datos.loc[datos.Unidades_Edaficas == 48, "Unidades_Edaficas"] = "Codigo_48"
    datos.loc[datos.Unidades_Edaficas == 23, "Unidades_Edaficas"] = "Codigo_23"
    datos.loc[datos.Unidades_Edaficas == 22, "Unidades_Edaficas"] = "Codigo_22"

    # Redondeo de los datos
    datos = datos.round(4)
    
    # En este caso se eliminan las cárcavas de 2011 no se van a utilizar
    del datos['Carcavas_2011']
    
    # Label Encoder
    # Damos a los valores categóricos como categoría
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    # Aplicamos label encoder par aque puedan ser tratados por el modelo
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col))    

    # Devolvemos los datos tratados
    return datos

"""
Las funciones carga de datos y tratamiento de datos son semejantes a las documentadas anteriormente
con pequeñas diferencias en la carga y en los tratamientos.
"""
def cargar_datos_Berrueco(path):
    print("\n-- Cargando datos Berrueco --")
    dir = path
    ficheros=[]
    nombre=[]
    
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            ficheros.append(os.path.join(dir, file))
            nombre.append(file.split('.')[0])
            
    sorted(ficheros)
    sorted(nombre)
    
    datos = pd.DataFrame(columns=nombre)
    
    tif = gdal.Open(ficheros[1])
    tif_band = tif.GetRasterBand(1)
    x = tif_band.XSize
    y = tif_band.YSize
    
    contador = 0
    for f in ficheros:
        n = nombre[contador]
        MDT = gdal.Open(f)

        # Patrones sin datos se incluyen como -1
        MDT.GetRasterBand(1).SetNoDataValue(-1)

        # Pasamos los datos a float
        band = MDT.GetRasterBand(1).ReadAsArray().astype(float)

        # Se desechan las dos primeras filas y columnas debido a calculos de borde
        band = band[1:y, 1:x]

        # Reconstruimos el array con el tamaño de filas x columnas
        x_recorte = x - 1
        y_recorte = y - 1
        band = np.reshape(band, x_recorte*y_recorte)

        # Estos serán nuestros datos
        datos[n] = band

        contador = contador + 1
       
    return datos

def tratamiento_datos_Berrueco(datos):
    print("\n-- Tratamiento datos Berrueco --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
    print("Número de datos despues del tratamiento ", datos.size)
    
    datos = datos.round(4) 
    
    # Geología
    datos.loc[datos.Geologia == 9001, "Geologia"] = "Codigo_9001"
    datos.loc[datos.Geologia == 9004, "Geologia"] = "Codigo_9004"
    datos.loc[datos.Geologia == 9103, "Geologia"] = "Codigo_9103"
    datos.loc[datos.Geologia == 9132, "Geologia"] = "Codigo_9132"
    datos.loc[datos.Geologia == 9133, "Geologia"] = "Codigo_9133"
    datos.loc[datos.Geologia == 9134, "Geologia"] = "Codigo_9134"
    datos.loc[datos.Geologia == 9201, "Geologia"] = "Codigo_9201"
    datos.loc[datos.Geologia == 9202, "Geologia"] = "Codigo_9202"
    
    # Unidades edáficas
    datos.loc[datos.Unidades_Edaficas == 1, "Unidades_Edaficas"] = "Codigo_23"
    datos.loc[datos.Unidades_Edaficas == 2, "Unidades_Edaficas"] = "Codigo_49"
    datos.loc[datos.Unidades_Edaficas == 3, "Unidades_Edaficas"] = "Codigo_11"
    datos.loc[datos.Unidades_Edaficas == 4, "Unidades_Edaficas"] = "Codigo_48"
    datos.loc[datos.Unidades_Edaficas == 5, "Unidades_Edaficas"] = "Codigo_21"
    datos.loc[datos.Unidades_Edaficas == 6, "Unidades_Edaficas"] = "Codigo_13"
    
    # Usos del suelo
    datos.loc[datos.Usos_Del_Suelo == 1, "Usos_Del_Suelo"] = "Tejido_urbano"
    datos.loc[datos.Usos_Del_Suelo == 2, "Usos_Del_Suelo"] = "Olivares"
    datos.loc[datos.Usos_Del_Suelo == 3, "Usos_Del_Suelo"] = "Cultivos_permanentes"
    datos.loc[datos.Usos_Del_Suelo == 4, "Usos_Del_Suelo"] = "Pastizales"

    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col)) 

    return datos

"""
En esta carga de datos hay una diferencia, al igual que en Rentillas

@resume: Observar código
"""
def cargar_datos_Lupion(path):
    print("\n-- Cargando datos Lupion --")
    dir = path
    ficheros=[]
    nombre=[]
    
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            ficheros.append(os.path.join(dir, file))
            nombre.append(file.split('.')[0])
            
    sorted(ficheros)
    sorted(nombre)
    
    datos = pd.DataFrame(columns=nombre)
    
    tif = gdal.Open(ficheros[1])
    tif_band = tif.GetRasterBand(1)
    x = tif_band.XSize
    y = tif_band.YSize
    
    contador = 0
    for f in ficheros:
        n = nombre[contador]
        MDT = gdal.Open(f)

        # Patrones sin datos se incluyen como -1
        MDT.GetRasterBand(1).SetNoDataValue(-1)

        # Pasamos los datos a float
        band = MDT.GetRasterBand(1).ReadAsArray().astype(float)

        """
        Como algunas de las capas no coinciden en las dimensiones proporcionadas,
        se recortan los bordes dependiendo de estas dimensiones para crear un estandar entre
        las variables
        """
        if(n == "Carcavas" or n == "Lupi_11_9999"): 
            band = band[:, 1:x]
        elif (n == "Curvatura_Perfil" or n == "Curvatura_Plana" or n == "Distancia_Carreteras"
              or n == "Orientaciones" or n == "Pendiente" or n == "Unidades_Edaficas" or n == "Usos_Del_Suelo"):
            band = band[:, :]
        else:
            band = band[1:y, 1:x] 

        # Reconstruimos el array con el tamaño de filas x columnas
        x_recorte = x - 1
        y_recorte = y - 1
        band = np.reshape(band, x_recorte*y_recorte)

        # Estos serán nuestros datos
        datos[n] = band

        contador = contador + 1
    
    return datos

def tratamiento_datos_Lupion(datos):
    print("\n-- Tratamiento datos Lupion --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Lupi_11_9999'] != -9999]
    datos = datos[datos['Altitud'] >= 0]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbonatos'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Carcavas'] != 255]
    datos = datos[datos['Curvatura_Perfil'] != -3.4028230607370965e+38]
    datos = datos[datos['Distancia_Carreteras'] >= 0]
    datos = datos[datos['Orientaciones'] >= 0]
    datos = datos[datos['Overland_Flow_Distance'] != -99999.0]
    
    # Geología
    datos.loc[datos.Geologia == 8996, "Geologia"] = "Codigo_8996"
    datos.loc[datos.Geologia == 9000, "Geologia"] = "Codigo_9000"
    datos.loc[datos.Geologia == 9001, "Geologia"] = "Codigo_9001"
    datos.loc[datos.Geologia == 9002, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Geologia == 9003, "Geologia"] = "Codigo_9003"
    datos.loc[datos.Geologia == 9004, "Geologia"] = "Codigo_9004"
    datos.loc[datos.Geologia == 9133, "Geologia"] = "Codigo_9133"
    datos.loc[datos.Geologia == 9134, "Geologia"] = "Codigo_9134"
    
    # Usos del suelo
    datos.loc[datos.Usos_Del_Suelo == 1, "Usos_Del_Suelo"] = "Tejido_urbano"
    datos.loc[datos.Usos_Del_Suelo == 2, "Usos_Del_Suelo"] = "Labor_secano"
    datos.loc[datos.Usos_Del_Suelo == 3, "Usos_Del_Suelo"] = "Tierras_regadas"
    datos.loc[datos.Usos_Del_Suelo == 4, "Usos_Del_Suelo"] = "Frutales"
    datos.loc[datos.Usos_Del_Suelo == 5, "Usos_Del_Suelo"] = "Olivares"
    datos.loc[datos.Usos_Del_Suelo == 6, "Usos_Del_Suelo"] = "Cultivos_permanentes"
    datos.loc[datos.Usos_Del_Suelo == 7, "Usos_Del_Suelo"] = "Mosaicos_cultivos"
    datos.loc[datos.Usos_Del_Suelo == 8, "Usos_Del_Suelo"] = "Vegetacion"
    datos.loc[datos.Usos_Del_Suelo == 9, "Usos_Del_Suelo"] = "Cursos_agua"
    
    # Unidades edaficas
    datos.loc[datos.Unidades_Edaficas == 1, "Unidades_Edaficas"] = "Codigo_48"
    datos.loc[datos.Unidades_Edaficas == 2, "Unidades_Edaficas"] = "Codigo_42"
    datos.loc[datos.Unidades_Edaficas == 3, "Unidades_Edaficas"] = "Codigo_44"
    datos.loc[datos.Unidades_Edaficas == 4, "Unidades_Edaficas"] = "Codigo_58"
    datos.loc[datos.Unidades_Edaficas == 5, "Unidades_Edaficas"] = "Codigo_23"
    datos.loc[datos.Unidades_Edaficas == 6, "Unidades_Edaficas"] = "Codigo_2"
    
    print("Número de datos despues del tratamiento ", datos.size)
    
    del datos['Lupi_11_9999']
    datos = datos.round(4) 
    
    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col)) 
    
    return datos

def cargar_datos_Rentillas(path):
    print("\n-- Cargando datos Rentillas --")
    dir = path
    ficheros=[]
    nombre=[]
    
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            ficheros.append(os.path.join(dir, file))
            nombre.append(file.split('.')[0])
            
    sorted(ficheros)
    sorted(nombre)
    
    datos = pd.DataFrame(columns=nombre)
    datos_sin_tocar = pd.DataFrame(columns=nombre)
    datos_para_dibujado = pd.DataFrame(columns=nombre)
    
    tif = gdal.Open(ficheros[0])
    tif_band = tif.GetRasterBand(1)
    x = tif_band.XSize
    y = tif_band.YSize
    
    contador = 0
    for f in ficheros:
        n = nombre[contador]
        MDT = gdal.Open(f)
        
        # Patrones sin datos se incluyen como -1
        MDT.GetRasterBand(1).SetNoDataValue(-1)
        
        # Pasamos los datos a float
        band = MDT.GetRasterBand(1).ReadAsArray().astype(float)
        
        # Se desechan las dos primeras filas y columnas debido a calculos de borde
        if(n == "Distancia_Carreteras" or n == "Geologia" or n == "Orientaciones" or
           n == "Pendiente" or n == "Unidades_Edaficas" or n == "Usos_Del_Suelo"): 
            band = band[1:y, :] 
        elif(n == "Overland_Flow_Distance"):
            band = band[:, :]
        else:
            band = band[1:y, 1:x]      
       
        x_recorte = x - 1
        y_recorte = y - 1
        
        # Reconstruimos el array con el tamaño de filas x columnas
        band = np.reshape(band, x_recorte*y_recorte)
        
        # Estos serán nuestros datos
        datos[n] = band
        datos_sin_tocar[n] = band
        datos_para_dibujado[n] = band

        contador = contador + 1
    
    #return datos, datos_para_dibujado, datos_sin_tocar, x, y
    return datos

def tratamiento_datos_Rentillas(datos):
    print("\n-- Tratamiento datos Rentillas --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Carcavas'] != -9999]
    datos = datos[datos['Altitud'] >= 0 ]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbonatos'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Distancia_Carreteras'] >= 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
    # Geología
    datos.loc[datos.Geologia == 9000, "Geologia"] = "Codigo_9000"
    datos.loc[datos.Geologia == 9001, "Geologia"] = "Codigo_9001"
    datos.loc[datos.Geologia == 9002, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Geologia == 9003, "Geologia"] = "Codigo_9003"
    datos.loc[datos.Geologia == 9101, "Geologia"] = "Codigo_9101"
    datos.loc[datos.Geologia == 9102, "Geologia"] = "Codigo_9102"
    datos.loc[datos.Geologia == 9133, "Geologia"] = "Codigo_9133"
    datos.loc[datos.Geologia == 9134, "Geologia"] = "Codigo_9134"
    
    # Usos del suelo
    datos.loc[datos.Usos_Del_Suelo == 1, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Usos_Del_Suelo == 2, "Usos_Del_Suelo"] = "Mosaicos_cultivos"
    datos.loc[datos.Usos_Del_Suelo == 3, "Usos_Del_Suelo"] = "Olivares"
    datos.loc[datos.Usos_Del_Suelo == 4, "Usos_Del_Suelo"] = "Tierras_regadas"
    datos.loc[datos.Usos_Del_Suelo == 5, "Usos_Del_Suelo"] = "Labor_secano"
    
    # Unidades edaficas
    datos.loc[datos.Unidades_Edaficas == 1, "Unidades_Edaficas"] = "Codigo_47"
    datos.loc[datos.Unidades_Edaficas == 2, "Unidades_Edaficas"] = "Codigo_37"
    datos.loc[datos.Unidades_Edaficas == 3, "Unidades_Edaficas"] = "Codigo_48"
    datos.loc[datos.Unidades_Edaficas == 4, "Unidades_Edaficas"] = "Codigo_2"

    
    print("Número de datos despues del tratamiento ", datos.size)
    
    datos = datos.round(4) 
    
    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col)) 
    
    return datos

"""
-- Fin de las funciones de carga y tratamiento de datos --
"""

"""
Union de los datos de zonas diferentes en un mismo Dataframe

@param: datos_X -> datos de las zonas tratadas
@return: datos
"""
def union_datos(datos_SantoTome, datos_Berrueco, datos_Lupion, datos_Rentillas):
    print("\n-- Uniendo datos --")
    frames = [datos_SantoTome, datos_Berrueco, datos_Lupion, datos_Rentillas]
    datos = pd.concat(frames, ignore_index=True)
    return datos

"""
Creación de matriz de correlación entre las variables

@param: datos -> datos de los que quieres sacar la correlación de sus instancias/variables
"""
def matriz_correlacion(datos):
    print("-- Matriz de confusion --")
    corr_datos = datos.corr(method='pearson')
    plt.figure(figsize=(18, 16))
    sns.heatmap(corr_datos, annot=True)
    plt.show()

"""
Sacar por pantalla el VIF de las instancias/variables de los datos

@param: datos -> datos de los que quieres sacar el VIF de sus instancias/variables
"""
def vif(datos):

    print("\n-- Realizando VIF --\n")
    
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = datos.columns
    
    # Calculando VIF para cada feature
    vif_data["VIF"] = [variance_inflation_factor(datos.values, i)
                          for i in range(len(datos.columns))]
    
    print("\n")
    print(vif_data)

"""
Definición del modelo XGBoost que se utilizará

@param: learning rate -> ratio de aprendizaje
@param: depth -> profundidad de arbol
@param: estimators -> estimadores
@param: valor_desequilibrio -> valor para modelos desbalanceados
@return: modelo

Fuente: https://xgboost.readthedocs.io/en/stable/parameter.html
"""
def modelo_XGBoost(learning_rate, depth, estimators, valos_desequilibrio):
  modelo_xgb = xgb.XGBClassifier(use_label_encoder=False, 
                                 verbosity=0, 
                                 eta=learning_rate,
                                 max_depth=depth, 
                                 sampling_method='gradient_based',
                                 scale_pos_weight=valos_desequilibrio,
                                 n_estimators=estimators)
  # Devuelve el modelo
  return modelo_xgb  

"""
Definicion de la importancia de pesos que han tenido las variables en el modelo de aprendizaje

@param: modelo -> modelo entrenado
"""
def importancia_variables(modelo):
    plot_importance(modelo)
    plt.show()

"""
Obtención de True Positive, False Negative, False Positive y True Negative de una matriz
de confusión

@param: Y_test -> variable dependiente de testeo
@param: Y_pred -> variable dependiente predicha
@return: True Positive, False Negative, False Positive y True Negative
"""
def m_confusion(Y_test, Y_pred):
    cnf_matrix = confusion_matrix(Y_test, Y_pred, labels=[1,0])
    tp, fn, fp, tn = cnf_matrix.ravel()
    return tp, fn, fp, tn

"""
Obtención del coeficiente Kappa

@param: TP -> True Positive
@param: FP -> False Positive
@param: FN -> False Negative
@param: TN -> True Negative
@return: Kappa
"""
def kappa(TP, FP, FN, TN):
    N = TP+TN+FP+FN

    precision = (TP+TN)/(TP+TN+FP+FN)
    
    Pexp = ((TP+FN)*(TP+FP)+(FP+TN)*(FN+TN))/(N*N)
    
    Pobs = (TP+TN)/N
    
    k = (Pobs - Pexp)/(1 - Pexp)
    
    return k

"""
Devuelve el valor Kappa y el valor de aciertos positivos

@param: modelo -> modelo creado sin entrenar
@param: x_pred -> valores a predecir
@param: y_P    -> variable dependiente de los valores a predecir
@return: Kappa, True Positive, RMSE, ROC
"""
def validaciones_modelo(modelo, x_pred, y_P):
    
    y_pred = modelo.predict(x_pred)
    tp, fn, fp, tn = m_confusion(y_P, y_pred)
    k = kappa(tp, fp, fn, tn)
    
    rmse, roc = rmse_auc(modelo, x_pred, y_P, y_pred)

    return k, tp, rmse, roc

"""
Elimina de los datos las variables seleccionadas en la función 
(realizar manualmente en la funcion)

@param: datos -> datos con las variables a eliminar
@return: datos
"""
def eliminacion_variables(datos):
    
    del datos['Unidades_Edaficas']
    del datos['Limos']
    del datos['Arcillas']
    del datos['Arenas']
    del datos['Carbonatos']
    
    return datos

"""
Realiza RSME y Curve ROC

@param: modelo -> modelo entrenado (si quieres utilizar esta funcion a parte, descomentar primera linea)
@param: test_x -> valores a predecir (si utilizas la funcion a parte)
@param: test_y -> variable dependiente de los valores predichos
@param: pred   -> resultado de la predicción del modelo
@return: rmse y roc
"""
def rmse_auc(modelo, test_x, test_y, pred):
    #pred = modelo.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    
    y_pred = modelo.predict_proba(test_x)[:,1]
    roc = roc_auc_score(test_y,y_pred)
    
    return rmse, roc

"""
Validación cruzada en 10 splits

@param: x -> variables independientes
@param: y -> variable/s dependiente/s
@param: modelo -> modelo definido
@return: resultados
"""
def validacion_cruzada(x, y, modelo):
    print("\n-- Validacion cruzada --\n")
    kfold = StratifiedKFold(n_splits=10)
    
    # Resultados que quieres observar en la validacion
    scoring = {'mse': 'neg_mean_squared_error',
               'precision_desbalanceada': 'balanced_accuracy',
               'precision': 'accuracy',
               'roc_auc': 'roc_auc',
               'f1': 'f1'}
    
    resultados = cross_validate(modelo, x, y, cv=kfold, scoring=scoring, return_train_score=True)
    
    return resultados


# -- PARAMETROS -- #
"""
Parametros utilizados para los diferentes modelos a crear, desde aquí se pueden
cambiar las variables que utiliza

- Modelo 1: Learning rate: 0.10, Profundidad 6 y número de estimadores 150
- Modelo 2: Learning rate: 0.25, Profundidad 10 y número de estimadores 250
"""

eta = 0.1
profundidad = 6
estimadores = 150

# -- MAIN -- #
"""
--- IMPORTANTE ---

El programa no sigue una secuencialidad. Algunas partes vendrán detalladas para ser opcionalmente
comentadas, oprimidas o utilizadas

--- IMPORTANTE ---
"""

# Cargamos los datos y tratamiento de -- SANTO TOME --
dir_SantoTome = "Raster/SantoTome_final"
datos_SantoTome = cargar_datos_SantoTome(dir_SantoTome)
datos_SantoTome = tratamiento_datos_SantoTome(datos_SantoTome)


# Cargamos los datos y tratamiento de -- BERRUECO -- 
dir_Berrueco = "Raster/Berrueco_final"
datos_Berrueco = cargar_datos_Berrueco(dir_Berrueco)
datos_Berrueco = tratamiento_datos_Berrueco(datos_Berrueco)


# Cargamos los datos y tratamiento de -- LUPION --
dir_Lupion = "Raster/Lupion_final"
datos_Lupion = cargar_datos_Lupion(dir_Lupion)
datos_Lupion = tratamiento_datos_Lupion(datos_Lupion)


# Cargamos los datos y tratamiento de -- RENTILLAS --
dir_Rentillas = "Raster/Rentillas_final"
datos_Rentillas = cargar_datos_Rentillas(dir_Rentillas)
datos_Rentillas = tratamiento_datos_Rentillas(datos_Rentillas)


# Union de datos
datos = union_datos(datos_SantoTome, datos_Berrueco, datos_Lupion, datos_Rentillas)


# Liberacion de memoria
del datos_SantoTome
del datos_Berrueco
del datos_Lupion
del datos_Rentillas


# Eliminacion de variables
datos = eliminacion_variables(datos)


# Matriz de correlacion de datos
matriz_correlacion(datos)


# Correlacion a través de VIF
vif(datos)


# Separacion de datos en X e Y
Y = datos.Carcavas
datos = datos.drop(['Carcavas'], axis=1)
X = datos


# Dividimos el dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)
valos_desequilibrio = math.sqrt(Y.value_counts()[0]/Y.value_counts()[1])

"""
-- IMPORTANTE --
Descomentar esta parte si tienes ya un modelo guardado, no sería necesario la creación
ni entrenamiento de otro modelo
"""
# Cargar modelos
# modelo = pickle.load(open("Modelos/modelo_general_1_variables", "rb"))


# Realizacion de modelo
print("\n-- Creando modelo --")
modelo = modelo_XGBoost(eta, profundidad, estimadores, valos_desequilibrio)


# Validacion cruzada -- INICIO --
resultados = validacion_cruzada(X, Y, modelo)
print(resultados.keys())

print("\n-- Parametros testeo --\n")
print("Precision desbalanceada: ", resultados['test_precision_desbalanceada'].mean())
print("Precision: ", resultados['test_precision'].mean())
print("MSE: ", resultados['test_mse'].mean())
mse = - resultados['test_mse'].mean()
print("RMSE: ", math.sqrt(mse))
print("ROC: ", resultados['test_roc_auc'].mean())
print("F1: ", resultados['test_f1'].mean())

print("\n-- Parametros entrenamiento --\n")
print("Precision desbalanceada: ", resultados['train_precision_desbalanceada'].mean())
print("Precision: ", resultados['train_precision'].mean())
print("MSE: ", resultados['train_mse'].mean())
mse = - resultados['train_mse'].mean()
print("RMSE: ", math.sqrt(mse))
print("ROC: ", resultados['train_roc_auc'].mean())
print("F1: ", resultados['train_f1'].mean())
# Validacion cruzada -- FINAL --


"""
-- IMPORTANTE --
Con la validación cruzada ya has entrenado el modelo el numero de splits correspondientes
por lo tanto no necesitas utilizar otro entrenamiento del modelo, si quieres entrenarlo a parte
comentar todo el apartado de la validación cruzada

-- IMPORTANTE -- 
Si has realizado la validación cruzada puedes comentar todo el código que hay a partir de aquí
"""

# Entrenamiento del modelo
print("\n-- Entrenando modelo --")
modelo.fit(X_train, Y_train)


"""
-- IMPORTANTE --
Esto guarda el modelo entrenado, con la validación cruzada esto no sirve, descomentar si se quiere
guardar el modelo y se ha utilizado -> modelo.fit
"""
# Guardamos modelo
# pickle.dump(modelo, open("Modelos/modelo_general_2_variablesxxxx", "wb"))

# Feature important
importancia_variables(modelo)

# Validacion
print("\n-- Validacion... --\n")
k, tp, rmse_valor, roc = validaciones_modelo(modelo, X_test, Y_test)
print("Kappa: ", k)
print("RMSE: ", rmse_valor, "ROC ", roc)


