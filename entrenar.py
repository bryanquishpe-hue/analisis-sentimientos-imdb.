
import os
import re
import html
import json
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Crear carpetas necesarias
os.makedirs("modelo", exist_ok=True)
os.makedirs("resultados", exist_ok=True)


# LIMPIAR TEXTOS
def limpiar_texto(texto):

    texto = html.unescape(str(texto))

    texto = re.sub(r"<.*?>", " ", texto)

    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)

    texto = texto.lower()

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


# ENTRENAMIENTO DEL MODELO
def entrenar_modelo(max_features=5000, C=1.0):

    print("\nINICIANDO ENTRENAMIENTO...")

    # Cargar dataset
    ruta = "dataset/IMDB Dataset.csv"

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No se encuentra dataset/IMDB Dataset.csv"
        )

    df = pd.read_csv(ruta)

    print("Registros originales:", len(df))

    # Verificar columnas
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("Columnas incorrectas en el dataset")

    # Eliminar datos vacíos y duplicados
    df = df.dropna(subset=["review", "sentiment"])

    df = df.drop_duplicates(subset=["review"])

    # Limpiar texto
    df["review"] = df["review"].apply(limpiar_texto)

    # Variables
    X = df["review"]

    y = df["sentiment"].str.lower().map({
        "positive": 1,
        "negative": 0
    })

    # Eliminar etiquetas inválidas
    validos = y.notna()

    X = X[validos]
    y = y[validos].astype(int)

    print("Registros válidos:", len(df))

    # GRÁFICA DE DISTRIBUCIÓN
    plt.figure(figsize=(7, 5))

    sns.countplot(
        x=y.map({
            0: "Negativo",
            1: "Positivo"
        })
    )

    plt.title("Distribución de sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad de reseñas")

    plt.tight_layout()

    plt.savefig("resultados/distribucion.png")

    plt.close()

    # DIVIDIR DATOS
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Datos de entrenamiento:", len(X_train))
    print("Datos de prueba:", len(X_test))

    # TF-IDF
    vectorizador = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_tfidf = vectorizador.fit_transform(X_train)

    X_test_tfidf = vectorizador.transform(X_test)

    # MODELO DE REGRESIÓN LOGÍSTICA
    modelo = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42
    )

    modelo.fit(X_train_tfidf, y_train)

    print("Modelo entrenado correctamente.")

    # PREDICCIONES
    predicciones = modelo.predict(X_test_tfidf)

    # MÉTRICAS
    metricas = {

        "accuracy": accuracy_score(
            y_test, predicciones
        ),

        "precision": precision_score(
            y_test, predicciones
        ),

        "recall": recall_score(
            y_test, predicciones
        ),

        "f1_score": f1_score(
            y_test, predicciones
        )
    }

    print("\n===== RESULTADOS DEL MODELO =====")

    for nombre, valor in metricas.items():

        print(f"{nombre}: {valor:.4f}")

    print("\nREPORTE DE CLASIFICACIÓN")

    print(
        classification_report(
            y_test,
            predicciones,
            target_names=[
                "Negativo",
                "Positivo"
            ]
        )
    )

    # MATRIZ DE CONFUSIÓN
    matriz = confusion_matrix(
        y_test,
        predicciones
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Negativo",
            "Positivo"
        ],
        yticklabels=[
            "Negativo",
            "Positivo"
        ]
    )

    plt.title("Matriz de confusión")

    plt.xlabel("Predicción")

    plt.ylabel("Valor real")

    plt.tight_layout()

    plt.savefig(
        "resultados/matriz_confusion.png"
    )

    plt.close()

    # GRÁFICA DE MÉTRICAS
    nombres = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    valores = [
        metricas["accuracy"],
        metricas["precision"],
        metricas["recall"],
        metricas["f1_score"]
    ]

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=nombres,
        y=valores
    )

    plt.ylim(0, 1)

    plt.title("Métricas del modelo")

    plt.xlabel("Métricas")

    plt.ylabel("Rendimiento")

    plt.tight_layout()

    plt.savefig(
        "resultados/metricas.png"
    )

    plt.close()

    # GUARDAR MODELO
    joblib.dump(
        modelo,
        "modelo/modelo_sentimientos.pkl"
    )

    joblib.dump(
        vectorizador,
        "modelo/vectorizador.pkl"
    )

    # GUARDAR MÉTRICAS EN JSON
    resultado = {

        "mensaje": "Entrenamiento realizado correctamente",

        "registros": int(len(X)),

        "entrenamiento": int(len(X_train)),

        "prueba": int(len(X_test)),

        "metricas": {
            nombre: float(valor)
            for nombre, valor in metricas.items()
        },

        "matriz_confusion": matriz.tolist()
    }

    with open(
        "resultados/metricas.json",
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultado,
            archivo,
            indent=4
        )

    print("\nMODELO GUARDADO CORRECTAMENTE.")

    return resultado
