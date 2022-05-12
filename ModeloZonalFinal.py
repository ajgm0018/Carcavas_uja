# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:58:50 2022

@author: Alberto Jose Gutierrez Megias
"""

# -- LIBRERIAS Y OPCIONES -- #

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



# -- FUNCIONES - #

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

def tratamiento_datos_sin_tocar_SantoTome(datos):
    print("\n-- Tratamiento datos sin tocar Santo Tome --\n")

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
    print("\n-- Tratamiento datos sin tocar Berrueco --\n")

    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
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
    print("\n-- Tratamiento datos sin tocar Lupion --\n")

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
    print("\n-- Tratamiento datos sin tocar Rentillas --\n")

    datos = datos[datos['Carcavas'] != -9999]
    datos = datos[datos['Altitud'] >= 0 ]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbonatos'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Distancia_Carreteras'] >= 0]
    datos = datos[datos['Orientaciones'] >= 0]
    
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

def union_datos(datos_1, datos_2, datos_3):
    print("\n-- Uniendo datos --")
    frames = [datos_1, datos_2, datos_3]
    datos = pd.concat(frames, ignore_index=True)
    return datos

def matriz_correlacion(datos):
    print("-- Matriz de confusion --")
    corr_datos = datos.corr(method='pearson')
    plt.figure(figsize=(18, 16))
    sns.heatmap(corr_datos, annot=True)
    plt.show()

def vif(datos):

    print("-- Realizando VIF --\n")
    
    """
    # Mapeamos
    datos['Geologia'] = datos['Geologia'].map({
        'Codigo_9002':0, 'Codigo_9133':1, 'Codigo_9001':2, 'Codigo_9330':3,
        'Codigo_9131':4, 'Codigo_9132':5, 'Codigo_9201':6, 'Codigo_9103':7,
        'Codigo_9202':8, 'Codigo_9004':9, 'Codigo_9134':10, 'Codigo_9000':11,
        'Codigo_9003':12, 'Codigo_8996':13, 'Codigo_9102':14})
    
    datos['Usos_Del_Suelo'] = datos['Usos_Del_Suelo'].map({
        'Tierras_regadas':0, 'Olivares':1, 'Labor_secano':2, 'Cultivos_permanentes':3,
        'Tejido_urbano':4, 'Pastizales':5, 'Mosaicos_cultivos':6, 'Cursos_agua':7,
        'Frutales':8})
    
    datos['Unidades_Edaficas'] = datos['Unidades_Edaficas'].map({
        'Codigo_22':0, 'Codigo_48':1, 'Codigo_44':2, 'Codigo_23':3,
        'Codigo_13':4, 'Codigo_21':5, 'Codigo_11':6, 'Codigo_49':7,
        'Codigo_2':8, 'Codigo_42':9, 'Codigo_58':10, 'Codigo_47':11,
        'Codigo_37':12})
    """
    
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = datos.columns
    
    # Calculando VIF para cada feature
    vif_data["VIF"] = [variance_inflation_factor(datos.values, i)
                          for i in range(len(datos.columns))]
    
    print("\n")
    print(vif_data)

def modelo_XGBoost(learning_rate, depth, estimators, valos_desequilibrio):
  modelo_xgb = xgb.XGBClassifier(use_label_encoder=False, 
                                 verbosity=0, 
                                 eta=learning_rate,
                                 max_depth=depth, 
                                 sampling_method='gradient_based',
                                 scale_pos_weight=valos_desequilibrio,
                                 n_estimators=estimators)
  return modelo_xgb  

def importancia_variables(modelo):
    plot_importance(modelo)
    plt.show()

def m_confusion(Y_test, Y_pred):
    cnf_matrix = confusion_matrix(Y_test, Y_pred, labels=[1,0])
    tp, fn, fp, tn = cnf_matrix.ravel()
    return tp, fn, fp, tn

def kappa(TP, FP, FN, TN):
    N = TP+TN+FP+FN

    precision = (TP+TN)/(TP+TN+FP+FN)
    
    Pexp = ((TP+FN)*(TP+FP)+(FP+TN)*(FN+TN))/(N*N)
    
    Pobs = (TP+TN)/N
    
    k = (Pobs - Pexp)/(1 - Pexp)
    
    return k

def validaciones_modelo(modelo, x_pred, y_P):
    
    y_pred = modelo.predict(x_pred)
    tp, fn, fp, tn = m_confusion(y_P, y_pred)
    k = kappa(tp, fp, fn, tn)

    return k, tp

def eliminacion_variables(datos):
    
    del datos['Unidades_Edaficas']
    del datos['Stream_Power_Index']
    del datos['Limos']
    del datos['Arcillas']
    del datos['Arenas']
    del datos['Carbonatos']
    
    return datos

def dibujo_mapa(zona, prediccion_prob, datos_para_dibujado, x, y):
    
    print("\n-- Realizando mapa de susceptiblidad --")
    
    # Realizacion del array de dibujado
    total = (prediccion_prob.size / 2)
    array_color = [0] * int(total)
    
    # array color: 1977628
    
    for i in range(int(total)):
        array_color[i] = prediccion_prob[i][1]
        
    if(zona == "Rentillas"):
        for i in datos_para_dibujado.index:
            if datos_para_dibujado["Carcavas"][i] == -9999 or datos_para_dibujado["Altitud"][i] < 0 or datos_para_dibujado["Distancia_Carreteras"][i] < 0 or datos_para_dibujado["Arcillas"][i] < 0 or datos_para_dibujado["Carbonatos"][i] < 0 or datos_para_dibujado["Carbono_Organico"][i] < 0 or datos_para_dibujado["Orientaciones"][i] < 0 :
                array_color[i] = 0
                
    # Comiendo de dibujo
    SalidaDibujo = np.reshape(array_color, (y-1,x-1))
    
    plt.rcParams['image.cmap'] = 'jet'
    plt.figure(figsize=(14,14))
    plt.imshow(SalidaDibujo, vmin=0, vmax=1)

# -- PARAMETROS -- #

eta = 0.1
profundidad = 6
estimadores = 150

# -- MAIN -- #

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


# Liberacion de memoria (Menos del que se quiere predecir)
del datos_SantoTome
del datos_Berrueco
del datos_Lupion
#del datos_Rentillas


# Eliminacion de variables
datos = eliminacion_variables(datos)
datos_sin_tocar_Rentillas = eliminacion_variables(datos_sin_tocar_Rentillas)


# Separacion de datos en X e Y
Y = datos.Carcavas
datos = datos.drop(['Carcavas'], axis=1)
datos_sin_tocar_Rentillas = datos_sin_tocar_Rentillas.drop(['Carcavas'], axis=1)
X = datos 


# Valor de desequilibrio
valos_desequilibrio = math.sqrt(Y.value_counts()[0]/Y.value_counts()[1])


# Realizacion de modelo
print("\n-- Creando modelo --")
modelo = modelo_XGBoost(eta, profundidad, estimadores, valos_desequilibrio)
print("\n-- Entrenando modelo --")
modelo.fit(X, Y)


# Prediccion
print("\n-- Realizando predicción --")
prediccion_prob = modelo.predict_proba(datos_sin_tocar_Rentillas)


# Mapa de susceptibilidad
dibujo_mapa("Rentillas", prediccion_prob, datos_Rentillas_dibujado, x_Rentillas, y_Rentillas)


