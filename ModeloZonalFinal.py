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
import matplotlib as mpl

import xgboost as xgb
from xgboost import plot_importance
from xgboost import cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pickle


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
    
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            ficheros.append(os.path.join(dir, file))
            nombre.append(file.split('.')[0])
            
    sorted(ficheros)
    sorted(nombre)
    
    datos = pd.DataFrame(columns=nombre)
    datos_sin_tocar = pd.DataFrame(columns=nombre)
    datos_para_dibujado = pd.DataFrame(columns=nombre)
    
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
        datos_sin_tocar[n] = band
        datos_para_dibujado[n] = band

        contador = contador + 1
    
    return datos, datos_para_dibujado, datos_sin_tocar, x, y

"""
Tratamiemto de los datos

@param: datos -> datos a tratar, en este caso deben de ser de Santo Tomé
@return: datos
"""
def tratamiento_datos_SantoTome(datos):
    print("\n-- Tratamiento datos Santo Tome --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Carcavas'] != 128]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Distancia_Carreteras'] >= 0]
    datos = datos[datos['Geologia'] != 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
    print("Número de datos despues del tratamiento ", datos.size)   
    
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

    datos = datos.round(4)
    del datos['Carcavas_2011']
    
    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col))    

    return datos


"""
Este tratamiento de datos no elimina los datos basura, les de unos valores aleatorios.
Estos valores basura se convertirán en 0. Pero es necesario que no sean eliminados porque
son utilizados para diferentes funciones donde necesitan las mismas dimensiones, como por ejemplo,
dibujar un mapa de susceptibilidad

@param: datos -> datos a tratar, en este caso deben de ser de Santo Tomé
@return: datos
"""
def tratamiento_datos_sin_tocar_SantoTome(datos):
    print("\n-- Tratamiento datos sin tocar Santo Tome --")

    # Estos cambios dan igual, se eliminan en el mapa es por que sean string
    datos.loc[datos.Carcavas == 128, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carcavas == 128, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carcavas == 128, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Arcillas < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Arcillas < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Arcillas < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Distancia_Carreteras < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Distancia_Carreteras < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Distancia_Carreteras < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Geologia == 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Geologia == 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Geologia == 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Orientaciones < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Orientaciones < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Orientaciones < 0, "Unidades_Edaficas"] = "Codigo_22"
    
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

    datos = datos.round(4)
    del datos['Carcavas_2011']
    del datos['Carcavas']
    
    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col))    

    return datos

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
    datos_sin_tocar = pd.DataFrame(columns=nombre)
    datos_para_dibujado = pd.DataFrame(columns=nombre)
    
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
        datos_sin_tocar[n] = band
        datos_para_dibujado[n] = band

        contador = contador + 1
    
    return datos, datos_para_dibujado, datos_sin_tocar, x, y    

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

def tratamiento_datos_sin_tocar_Berrueco(datos):
    print("\n-- Tratamiento datos sin tocar Berrueco --")
    
    # Estos cambios dan igual, se eliminan en el mapa es por que sean string
    datos.loc[datos.Arcillas < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Arcillas < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Arcillas < 0, "Unidades_Edaficas"] = "Codigo_22"

    datos.loc[datos.Carbono_Organico < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carbono_Organico < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carbono_Organico < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Orientaciones < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Orientaciones < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Orientaciones < 0, "Unidades_Edaficas"] = "Codigo_22"
    
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

    del datos['Carcavas']

    # Label Encoder
    datos["Geologia"] = datos["Geologia"].astype("category")
    datos["Usos_Del_Suelo"] = datos["Usos_Del_Suelo"].astype("category")
    datos["Unidades_Edaficas"] = datos["Unidades_Edaficas"].astype("category")
    
    categorical_cols = ['Geologia', 'Usos_Del_Suelo', 'Unidades_Edaficas']
    
    le = LabelEncoder()
    
    datos[categorical_cols] = datos[categorical_cols].apply(lambda col: le.fit_transform(col)) 

    return datos

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
    datos_sin_tocar = pd.DataFrame(columns=nombre)
    datos_para_dibujado = pd.DataFrame(columns=nombre)
    
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
        datos_sin_tocar[n] = band
        datos_para_dibujado[n] = band

        contador = contador + 1
    
    return datos, datos_para_dibujado, datos_sin_tocar, x, y
    
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

def tratamiento_datos_sin_tocar_Lupion(datos):
    print("\n-- Tratamiento datos sin tocar Lupion --")
    
    # Estos cambios dan igual, se eliminan en el mapa es por que sean string
    datos.loc[datos.Lupi_11_9999 == -9999, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Lupi_11_9999 == -9999, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Lupi_11_9999 == -9999, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Altitud < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Altitud < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Altitud < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Arcillas < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Arcillas < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Arcillas < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Carbonatos < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carbonatos < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carbonatos < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Carbono_Organico < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carbono_Organico < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carbono_Organico < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Carcavas == 255, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carcavas == 255, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carcavas == 255, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Curvatura_Perfil == -3.4028230607370965e+38, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Curvatura_Perfil == -3.4028230607370965e+38, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Curvatura_Perfil == -3.4028230607370965e+38, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Distancia_Carreteras < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Distancia_Carreteras < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Distancia_Carreteras < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Orientaciones < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Orientaciones < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Orientaciones < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Overland_Flow_Distance == -99999.0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Overland_Flow_Distance == -99999.0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Overland_Flow_Distance == -99999.0, "Unidades_Edaficas"] = "Codigo_22"
    
    
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
    
    del datos['Lupi_11_9999']
    del datos['Carcavas']
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
    
    return datos, datos_para_dibujado, datos_sin_tocar, x, y
    
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

def tratamiento_datos_sin_tocar_Rentillas(datos):
    print("\n-- Tratamiento datos sin tocar Rentillas --")
    
    # Estos cambios dan igual, se eliminan en el mapa es por que sean string
    datos.loc[datos.Carcavas == -9999, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carcavas == -9999, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carcavas == -9999, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Altitud < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Altitud < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Altitud < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Arcillas < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Arcillas < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Arcillas < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Carbonatos < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carbonatos < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carbonatos < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Carbono_Organico < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Carbono_Organico < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Carbono_Organico < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Distancia_Carreteras < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Distancia_Carreteras < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Distancia_Carreteras < 0, "Unidades_Edaficas"] = "Codigo_22"
    
    datos.loc[datos.Orientaciones < 0, "Geologia"] = "Codigo_9002"
    datos.loc[datos.Orientaciones < 0, "Usos_Del_Suelo"] = "Cursos_agua"
    datos.loc[datos.Orientaciones < 0, "Unidades_Edaficas"] = "Codigo_22"
    
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
    
    datos = datos.round(4) 
    
    del datos['Carcavas']
    
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
def union_datos(datos_1, datos_2, datos_3):
    print("\n-- Uniendo datos --")
    frames = [datos_1, datos_2, datos_3]
    datos = pd.concat(frames, ignore_index=True)
    return datos

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
  return modelo_xgb  

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
Ademas muestra la Precisión, MSE, RMSE, ROC y F1 del modelo despues de predecir

@param: modelo -> modelo creado sin entrenar
@param: x_pred -> valores a predecir
@param: y_P    -> variable dependiente de los valores a predecir
@return: Kappa, True Positive
"""
def validaciones_modelo(modelo, x_pred, y_P):
    
    y_pred = modelo.predict(x_pred)
    tp, fn, fp, tn = m_confusion(y_P, y_pred)
    k = kappa(tp, fp, fn, tn)
    mse = mean_squared_error(y_P, y_pred)
    
    print("Precision: ", accuracy_score(y_P, y_pred))
    print("MSE: ", mse)
    print("RMSE: ", math.sqrt(mse))
    print("ROC: ", roc_auc_score(y_P, modelo.predict_proba(x_pred)[:, 1]))
    print("F1: ", f1_score(y_P, y_pred))

    return k, tp

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
Realizar mapa de susceptibilidad continuo a través de los valores de probabilidad de cada pixel de que
en ese mismo pixel haya o no una cárcava

Dependiendo de la zona, a cada uno de los valores basura que no fueron borrados para transformarlos en 0
ocurre en esta función indicado en un comentario. Esto se hace para realizar un mapa con una dimensión
determinada

@param: zona -> Lugar geográfico a dibujar mapa (Opciones: Rentillas, Lupion, Berrueco, SantoTome)
@param: prediccion_prob -> Probabilidades de la predicción ya realizada
@param: datos_para_dibujado -> datos utilizados para dibujar
@param: x -> Dimension X del mapa
@param: y -> Dimension Y del mapa
@return: Array con las probabilidades de los mapas
    
"""
def dibujo_mapa(zona, prediccion_prob, datos_para_dibujado, x, y):
    
    print("\n-- Realizando mapa de susceptiblidad --")
    
    # Realizacion del array de dibujado
    total = (prediccion_prob.size / 2)
    array_color = [0] * int(total)
    
    for i in range(int(total)):
        array_color[i] = prediccion_prob[i][1]
      
    # Aqui se realiza la transformación a ceros
    if(zona == "Rentillas"):
        for i in datos_para_dibujado.index:
            if datos_para_dibujado["Carcavas"][i] == -9999 or datos_para_dibujado["Altitud"][i] < 0 or datos_para_dibujado["Distancia_Carreteras"][i] < 0 or datos_para_dibujado["Carbono_Organico"][i] < 0 or datos_para_dibujado["Orientaciones"][i] < 0 or datos_para_dibujado["Carbonatos"][i] < 0 or datos_para_dibujado["Arcillas"][i] < 0:                
                array_color[i] = 0
    if(zona == "Lupion"):
        for i in datos_para_dibujado.index:
            if datos_para_dibujado["Carcavas"][i] == -255 or datos_para_dibujado["Altitud"][i] < 0 or datos_para_dibujado["Distancia_Carreteras"][i] < 0 or datos_para_dibujado["Carbono_Organico"][i] < 0 or datos_para_dibujado["Curvatura_Perfil"][i] == -3.4028230607370965e+38 or datos_para_dibujado["Distancia_Carreteras"][i] < 0 or datos_para_dibujado["Orientaciones"][i] < 0 or datos_para_dibujado["Overland_Flow_Distance"][i] < -99999.0 or datos_para_dibujado["Carbonatos"][i] < 0 or datos_para_dibujado["Arcillas"][i] < 0:                
                array_color[i] = 0
    if(zona == "Berrueco"):
        for i in datos_para_dibujado.index:
            if datos_para_dibujado["Carbono_Organico"][i] < 0 or datos_para_dibujado["Orientaciones"][i] < 0 or datos_para_dibujado["Arcillas"][i] < 0:                
                array_color[i] = 0
    if(zona == "SantoTome"):
        for i in datos_para_dibujado.index:
            if datos_para_dibujado["Carcavas"][i] == -128 or datos_para_dibujado["Distancia_Carreteras"][i] < 0 or datos_para_dibujado["Orientaciones"][i] < 0 or datos_para_dibujado["Arcillas"][i] < 0 or datos_para_dibujado["Geologia"][i] < 0:                
                array_color[i] = 0
                
    # Comiendo de dibujo
    SalidaDibujo = np.reshape(array_color, (y-1,x-1))
    
    plt.rcParams['image.cmap'] = 'jet'
    plt.figure(figsize=(14,14))
    plt.imshow(SalidaDibujo, vmin=0, vmax=1)
    
    return array_color

"""
Realizar mapa de susceptibilidad discreto

@param: prediccion -> Prediccion en probabilidad
@param: x -> Dimension X del mapa
@param: y -> Dimension Y del mapa
@return: Salida del dibujo (mapa de susceptibilidad)
"""
def dibujar_mapa_2(prediccion, x, y):

    # Rango de colores
    cmap = mpl.colors.ListedColormap(['green','limegreen','yellow','orange','darkgoldenrod','red'])
    # Rango numérico de probabilidades por color
    bins = [0.0, 0.05, 0.2, 0.4, 0.7, 0.8, 1.0]

    norm = mpl.colors.BoundaryNorm(boundaries=bins, ncolors=len(cmap.colors) )

    SalidaDibujo = np.reshape(prediccion, (y,x))
    
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    
    fig, ax = plt.subplots(figsize=(14,14))
    ax.pcolormesh(x, y, SalidaDibujo, cmap = cmap, norm=norm, shading='flat')

    fig.show()
    return SalidaDibujo

"""
Realiza un dibujo de la barra discreta
"""
def dibujarCustomBar():
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.colors.ListedColormap(['green','limegreen','yellow','orange','darkgoldenrod','red'])
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    bounds = [0.0, 0.05, 0.2, 0.4, 0.7, 0.8, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    boundaries=[0] + bounds + [13],
                                    extend='both',
                                    ticks=bounds,
                                    spacing='proportional',
                                    orientation='horizontal')
    fig.show()
 
"""
Proceso de creación de CSV para transformarlo a TIF en otro programa en R
"""
def preparacion_TIF(prediccion, x, y, nombre_archivo):
    ("\n-- Realizando copia preparada para TIF --")
    
    salida_preparada = np.reshape(prediccion, (y,x))
    
    array_zero = np.zeros(salida_preparada.shape[1])
    array_zero = array_zero + 1
    
    salida = np.vstack((array_zero, salida_preparada))
    
    array_zero_2 = np.zeros(salida.shape[0])
    salida = np.hstack((array_zero_2.reshape(len(array_zero_2), 1), salida))
    
    pd.DataFrame(salida).to_csv(nombre_archivo, index = False, header = False)



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
datos_SantoTome, datos_SantoTome_dibujado, datos_sin_tocar_SantoTome, x_SantoTome, y_SantoTome = cargar_datos_SantoTome(dir_SantoTome)
datos_SantoTome = tratamiento_datos_SantoTome(datos_SantoTome)
datos_sin_tocar_SantoTome = tratamiento_datos_sin_tocar_SantoTome(datos_sin_tocar_SantoTome)


# Cargamos los datos y tratamiento de -- BERRUECO -- 
dir_Berrueco = "Raster/Berrueco_final"
datos_Berrueco, datos_Berrueco_dibujado, datos_sin_tocar_Berrueco, x_Berrueco, y_Berrueco = cargar_datos_Berrueco(dir_Berrueco)
datos_Berrueco = tratamiento_datos_Berrueco(datos_Berrueco)
datos_sin_tocar_Berrueco = tratamiento_datos_sin_tocar_Berrueco(datos_sin_tocar_Berrueco)


# Cargamos los datos y tratamiento de -- LUPION --
dir_Lupion = "Raster/Lupion_final"
datos_Lupion, datos_Lupion_dibujado, datos_sin_tocar_Lupion, x_Lupion, y_Lupion = cargar_datos_Lupion(dir_Lupion)
datos_Lupion = tratamiento_datos_Lupion(datos_Lupion)
datos_sin_tocar_Lupion = tratamiento_datos_sin_tocar_Lupion(datos_sin_tocar_Lupion)


# Cargamos los datos y tratamiento de -- RENTILLAS --
dir_Rentillas = "Raster/Rentillas_final"
datos_Rentillas, datos_Rentillas_dibujado, datos_sin_tocar_Rentillas, x_Rentillas, y_Rentillas = cargar_datos_Rentillas(dir_Rentillas)
datos_Rentillas = tratamiento_datos_Rentillas(datos_Rentillas)
datos_sin_tocar_Rentillas = tratamiento_datos_sin_tocar_Rentillas(datos_sin_tocar_Rentillas)


# Union de datos
datos = union_datos(datos_SantoTome, datos_Berrueco, datos_Lupion)


# Eliminacion de variables
datos = eliminacion_variables(datos)

"""
-- IMPORTANTE --
Solo descomentar la zona a predecir
"""
# Rentillas
#datos_sin_tocar_Rentillas = eliminacion_variables(datos_sin_tocar_Rentillas)

# Lupion
#datos_sin_tocar_Lupion = eliminacion_variables(datos_sin_tocar_Lupion)

# Berrueco
#datos_sin_tocar_Berrueco = eliminacion_variables(datos_sin_tocar_Berrueco)

# Santo Tome
datos_sin_tocar_SantoTome = eliminacion_variables(datos_sin_tocar_SantoTome)


# Separacion de datos en X e Y
Y = datos.Carcavas
datos = datos.drop(['Carcavas'], axis=1)
X = datos 


# Valor de desequilibrio
valos_desequilibrio = math.sqrt(Y.value_counts()[0]/Y.value_counts()[1])



# Realizacion de modelo
print("\n-- Creando modelo --")
modelo = modelo_XGBoost(eta, profundidad, estimadores, valos_desequilibrio)
print("\n-- Entrenando modelo --")
modelo.fit(X, Y)

"""
-- IMPORTANTE --
Descomentar la carga del modelo si ya tienes un modelo generado, si haces esto, comentar todos
los pasos relacionados con la generacion del modelo anteriormente
"""
# Carga de modelo
# modelo = pickle.load(open("Modelos/modelo_general_1_variables", "rb"))


# Prediccion probabilidades
print("\n-- Realizando predicción probabilidad --")
prediccion_prob = modelo.predict_proba(datos_sin_tocar_SantoTome)


# Separacion para prediccion normal
datos_SantoTome = eliminacion_variables(datos_SantoTome)
Y_2 = datos_SantoTome.Carcavas
datos_SantoTome = datos_SantoTome.drop(['Carcavas'], axis=1)
X_2 = datos_SantoTome 


# Validaciones normales
print("\n-- Realizando predicción normal y validaciones --")
validaciones_modelo(modelo, X_2, Y_2)


# Mapa de susceptibilidad
array_color = dibujo_mapa("SantoTome", prediccion_prob, datos_SantoTome_dibujado, x_SantoTome, y_SantoTome)


# Mapa de susceptibilidad
dibujar_mapa_2(array_color, x_SantoTome-1, y_SantoTome-1)


# Custom bar
dibujarCustomBar()


# A CSV para pasar a TIF
preparacion_TIF(array_color, x_SantoTome-1, y_SantoTome-1, "CSV/SantoTome_Prob_General.csv")












