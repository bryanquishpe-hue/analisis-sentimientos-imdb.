
# Análisis de Sentimientos IMDB

## Descripción

Aplicación de inteligencia artificial desarrollada en Python
para clasificar reseñas de películas en sentimientos positivos
y negativos.

## Objetivo

Implementar un modelo de aprendizaje automático que permita
analizar opiniones de películas mediante procesamiento
de lenguaje natural.

## Dataset

IMDB Dataset of 50K Movie Reviews.

Fuente:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

El dataset contiene reseñas reales de películas
clasificadas como positivas y negativas.

## Tecnologías utilizadas

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- HTML
- CSS
- JavaScript

## Modelo de inteligencia artificial

Se utiliza TF-IDF para transformar los textos en
representaciones numéricas y Regresión Logística
para clasificar los sentimientos.

## Métricas

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de confusión

## Instalación

Crear entorno virtual:

python -m venv venv

Activar entorno virtual:

Windows PowerShell:

.\venv\Scripts\Activate.ps1

Instalar dependencias:

pip install -r requirements.txt

## Descargar dataset

Ejecutar:

python descargar_dataset.py

## Ejecutar aplicación

python app.py

Abrir en el navegador:

http://127.0.0.1:5000

## Funcionalidades

1. Entrenamiento del modelo.
2. Configuración de parámetros.
3. Predicción de sentimientos.
4. Visualización de métricas.
5. Generación de gráficas.
6. API REST con Flask.

## Bibliografía

Maas, A. L., et al. (2011).
Learning Word Vectors for Sentiment Analysis.
ACL.

Kaggle. IMDB Dataset of 50K Movie Reviews.

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Scikit-learn Documentation:
https://scikit-learn.org/

Flask Documentation:
https://flask.palletsprojects.com/
