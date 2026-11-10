
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

import os
import joblib

from entrenar import entrenar_modelo, limpiar_texto


app = Flask(__name__)

# Variables globales
modelo = None
vectorizador = None


# CARGAR MODELO GUARDADO
def cargar_modelo():

    global modelo, vectorizador

    ruta_modelo = "modelo/modelo_sentimientos.pkl"
    ruta_vectorizador = "modelo/vectorizador.pkl"

    if (
        os.path.exists(ruta_modelo)
        and os.path.exists(ruta_vectorizador)
    ):

        modelo = joblib.load(ruta_modelo)

        vectorizador = joblib.load(ruta_vectorizador)

        print("Modelo cargado correctamente.")


# PÁGINA PRINCIPAL
@app.route("/")
def inicio():

    return render_template("index.html")


# API PARA ENTRENAR
@app.route("/api/entrenar", methods=["POST"])
def entrenar():

    global modelo, vectorizador

    try:

        datos = request.get_json() or {}

        max_features = int(
            datos.get("max_features", 5000)
        )

        C = float(
            datos.get("C", 1.0)
        )

        if not 100 <= max_features <= 20000:
            return jsonify({
                "error": "Las características deben estar entre 100 y 20000."
            }), 400

        if not 0.01 <= C <= 10:
            return jsonify({
                "error": "El parámetro C debe estar entre 0.01 y 10."
            }), 400

        # Entrenar modelo
        resultado = entrenar_modelo(
            max_features=max_features,
            C=C
        )

        # Cargar modelo entrenado
        cargar_modelo()

        return jsonify(resultado)

    except Exception as e:

        print("Error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# API PARA PREDECIR
@app.route("/api/predecir", methods=["POST"])
def predecir():

    global modelo, vectorizador

    try:

        if modelo is None or vectorizador is None:
            cargar_modelo()

        if modelo is None:
            return jsonify({
                "error": "Primero debes entrenar el modelo."
            }), 400

        datos = request.get_json() or {}

        texto = datos.get("review", "").strip()

        if not texto:
            return jsonify({
                "error": "Debes escribir una reseña."
            }), 400

        # Limpiar texto
        texto_limpio = limpiar_texto(texto)

        if not texto_limpio:
            return jsonify({
                "error": "No se encontró texto válido."
            }), 400

        # Transformar texto
        texto_vectorizado = vectorizador.transform(
            [texto_limpio]
        )

        # Realizar predicción
        prediccion = int(
            modelo.predict(texto_vectorizado)[0]
        )

        probabilidades = modelo.predict_proba(
            texto_vectorizado
        )[0]

        sentimiento = (
            "Positivo" if prediccion == 1
            else "Negativo"
        )

        confianza = float(
            probabilidades[prediccion] * 100
        )

        return jsonify({

            "sentimiento": sentimiento,

            "confianza": round(confianza, 2),

            "probabilidad_positiva": round(
                float(probabilidades[1]) * 100, 2
            ),

            "probabilidad_negativa": round(
                float(probabilidades[0]) * 100, 2
            )

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# API PARA CONSULTAR MÉTRICAS
@app.route("/api/metricas", methods=["GET"])
def obtener_metricas():

    ruta = "resultados/metricas.json"

    if os.path.exists(ruta):

        return send_from_directory(
            "resultados",
            "metricas.json"
        )

    return jsonify({
        "error": "Todavía no existen métricas. Entrena el modelo."
    }), 404


# MOSTRAR GRÁFICAS
@app.route("/resultados/<path:archivo>")
def mostrar_resultados(archivo):

    return send_from_directory(
        "resultados",
        archivo
    )


# INICIAR SERVIDOR
if __name__ == "__main__":

    cargar_modelo()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
