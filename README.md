# Carcavas_uja

Codigo documentado: 

- ModeloGeneralFinal
- ModeloZonalFinal

Codigo para la realización de predicción de diferentes zonas geográficas en la Provincia de Jaén con datos Geoespaciales

# Para transformar un CSV generado en ModeloZonalFinal seguir los siguientes pasos en R

library(raster)

r <- raster(<archivo>.tif) (archivo ya existente)
  
matriz <- read.csv(URL, header = FALSE)
          
matriz_final <- matrix(unlist(matriz), ncol = 980, nrow = 1540)
  
r2 <- setValues(r, matriz_final )
      
writeRaster(r2, <directorio+nombre.tif>, format = 'GTiff')
