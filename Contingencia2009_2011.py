# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:32:12 2022

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
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder
import pickle


# -- FUNCIONES - #

def tabla_contingencia(carcavas_2009_pred, carcavas_2011_existentes):
    
    contador = 0
    numero_acierto = 0
    no_hay = 0
    
    for valor in carcavas_2009_pred:
        if valor == 1:
            if carcavas_2011_existentes.iloc[contador] == 1:
                numero_acierto += 1
            else:
                no_hay += 1
        contador += 1
    
    print("\nEn 2009 se predijo cárcavas de 2009:", numero_acierto, "aciertos")
    print("\nEn 2009 se predijo cárcavas de 2011 y no había:", no_hay, "fallos")  

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
    
    return datos

def tratamiento_datos_SantoTome(datos):
    print("\n-- Tratamiento datos Santo Tome --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Carcavas_2009'] != 128]
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
        if(n == "Carcavas_2009" or n == "Carcavas_2011"): 
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
    
    return datos
    
def tratamiento_datos_Lupion(datos):
    print("\n-- Tratamiento datos Lupion --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Carcavas_2011'] != -9999]
    datos = datos[datos['Altitud'] >= 0]
    datos = datos[datos['Arcillas'] >= 0]
    datos = datos[datos['Carbonatos'] >= 0]
    datos = datos[datos['Carbono_Organico'] >= 0]
    datos = datos[datos['Carcavas_2009'] != 255]
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
           n == "Pendiente" or n == "Unidades_Edaficas" or n == "Usos_Del_Suelo" or n == "Carcavas_2011"): 
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
    
    return datos
    
def tratamiento_datos_Rentillas(datos):
    print("\n-- Tratamiento datos Rentillas --\n")
    print("Número de datos antes del tratamiento ", datos.size)

    datos = datos[datos['Carcavas_2009'] != -9999]
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

def eliminacion_variables(datos):
    
    del datos['Unidades_Edaficas']
    del datos['Limos']
    del datos['Arcillas']
    del datos['Arenas']
    del datos['Carbonatos']
    
    return datos


# -- PARAMETROS -- #
"""
- Modelo 1: Learning rate: 0.10, Profundidad 6 y número de estimadores 150
- Modelo 2: Learning rate: 0.25, Profundidad 10 y número de estimadores 250
"""

eta = 0.1
profundidad = 6
estimadores = 150

# -- MAIN -- #

# Cargamos los datos y tratamiento de -- SANTO TOME --
dir_SantoTome = "Raster/SantoTome_Contingecia"
datos_SantoTome = cargar_datos_SantoTome(dir_SantoTome)
datos_SantoTome = tratamiento_datos_SantoTome(datos_SantoTome)


# Cargamos los datos y tratamiento de -- BERRUECO -- 
dir_Berrueco = "Raster/Berrueco_Contingencia"
datos_Berrueco = cargar_datos_Berrueco(dir_Berrueco)
datos_Berrueco = tratamiento_datos_Berrueco(datos_Berrueco)


# Cargamos los datos y tratamiento de -- LUPION --
dir_Lupion = "Raster/Lupion_Contingencia"
datos_Lupion = cargar_datos_Lupion(dir_Lupion)
datos_Lupion = tratamiento_datos_Lupion(datos_Lupion)


# Cargamos los datos y tratamiento de -- RENTILLAS --
dir_Rentillas = "Raster/Rentillas_Contingencia"
datos_Rentillas = cargar_datos_Rentillas(dir_Rentillas)
datos_Rentillas = tratamiento_datos_Rentillas(datos_Rentillas)


# Separamos las carcavas de 2011 de todas las zonas
datos_SantoTome = eliminacion_variables(datos_SantoTome)
Carcavas_2011_SantoTome = datos_SantoTome.Carcavas_2011
datos_SantoTome = datos_SantoTome.drop(['Carcavas_2011'], axis=1)
datos_SantoTome = datos_SantoTome.drop(['Carcavas_2009'], axis=1)

datos_Berrueco = eliminacion_variables(datos_Berrueco)
Carcavas_2011_Berrueco = datos_Berrueco.Carcavas_2011
datos_Berrueco = datos_Berrueco.drop(['Carcavas_2011'], axis=1)
datos_Berrueco = datos_Berrueco.drop(['Carcavas_2009'], axis=1)

datos_Lupion = eliminacion_variables(datos_Lupion)
Carcavas_2011_Lupion = datos_Lupion.Carcavas_2011
datos_Lupion = datos_Lupion.drop(['Carcavas_2011'], axis=1)
datos_Lupion = datos_Lupion.drop(['Carcavas_2009'], axis=1)


datos_Rentillas = eliminacion_variables(datos_Rentillas)
Carcavas_2011_Rentillas = datos_Rentillas.Carcavas_2011
datos_Rentillas = datos_Rentillas.drop(['Carcavas_2011'], axis=1)
datos_Rentillas = datos_Rentillas.drop(['Carcavas_2009'], axis=1)


# Carcagamos el modelo
modelo = pickle.load(open("Modelos/modelo_general_1_variables", "rb"))


# Prediccion
print("\n-- Predicciones --")
prediccion_SantoTome = modelo.predict(datos_SantoTome)
prediccion_Berrueco = modelo.predict(datos_Berrueco)
prediccion_Rentillas = modelo.predict(datos_Rentillas)
prediccion_Lupion = modelo.predict(datos_Lupion)


# Tabla
"""
print("\n-- Realizando tabla de Santo Tomé --")
tabla_contingencia(prediccion_SantoTome, Carcavas_2011_SantoTome)
print("\n-- Realizando tabla de Berrueco --")
tabla_contingencia(prediccion_Berrueco, Carcavas_2011_Berrueco)
print("\n-- Realizando tabla de Rentillas --")
tabla_contingencia(prediccion_Rentillas, Carcavas_2011_Rentillas)
print("\n-- Realizando tabla de Lupion --")
tabla_contingencia(prediccion_Lupion, Carcavas_2011_Lupion)
"""













