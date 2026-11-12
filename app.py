from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import joblib

from entrenar import entrenar_modelo, limpiar_texto

app = Flask(__name__)

# ==========================================
# RUTAS DE LOS ARCHIVOS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CARPETA_MODELO = os.path.join(BASE_DIR, "modelo")
CARPETA_RESULTADOS = os.path.join(BASE_DIR, "resultados")

RUTA_MODELO = os.path.join(CARPETA_MODELO, "modelo_sentimientos.pkl")
RUTA_VECTORIZADOR = os.path.join(CARPETA_MODELO, "vectorizador.pkl")

modelo = None
vectorizador = None


# ==========================================
# CARGAR MODELO GUARDADO
# ==========================================

def cargar_modelo():
    global modelo, vectorizador

    if os.path.exists(RUTA_MODELO) and os.path.exists(RUTA_VECTORIZADOR):
        modelo = joblib.load(RUTA_MODELO)
        vectorizador = joblib.load(RUTA_VECTORIZADOR)
        print("Modelo cargado correctamente.")
    else:
        print("Todavía no existe un modelo entrenado.")


# ==========================================
# PAGINAS PRINCIPALES
# ==========================================

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/analisis")
def analisis():
    return render_template("analisis.html")


@app.route("/resultados")
def resultados():
    return render_template("resultados.html")


# ==========================================
# API PARA ENTRENAR MODELO
# ==========================================

@app.route("/api/entrenar", methods=["POST"])
def entrenar():
    try:
        datos = request.get_json() or {}

        max_features = int(datos.get("max_features", 5000))
        C = float(datos.get("C", 1.0))

        if not 100 <= max_features <= 20000:
            return jsonify({
                "error": "Las características deben estar entre 100 y 20000."
            }), 400

        if not 0.01 <= C <= 10:
            return jsonify({
                "error": "El parámetro C debe estar entre 0.01 y 10."
            }), 400

        resultado = entrenar_modelo(max_features=max_features, C=C)

        cargar_modelo()

        return jsonify({
            "mensaje": "Entrenamiento realizado correctamente.",
            "metricas": resultado
        })

    except Exception as e:
        print("Error de entrenamiento:", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================================
# API PARA ANALIZAR UNA RESEÑA
# ==========================================

@app.route("/api/predecir", methods=["POST"])
def predecir():
    try:
        if modelo is None or vectorizador is None:
            cargar_modelo()

        if modelo is None or vectorizador is None:
            return jsonify({"error": "Primero debes entrenar el modelo."}), 400

        datos = request.get_json() or {}

        texto = (
            datos.get("review")
            or datos.get("resena")
            or datos.get("texto")
            or ""
        ).strip()

        if not texto:
            return jsonify({"error": "Debes escribir una reseña."}), 400

        texto_limpio = limpiar_texto(texto)

        if not texto_limpio:
            return jsonify({"error": "No se encontró texto válido."}), 400

        texto_vectorizado = vectorizador.transform([texto_limpio])

        prediccion = int(modelo.predict(texto_vectorizado)[0])
        probabilidades = modelo.predict_proba(texto_vectorizado)[0]

        sentimiento = "Positivo" if prediccion == 1 else "Negativo"
        confianza = float(probabilidades[prediccion] * 100)

        return jsonify({
            "sentimiento": sentimiento,
            "confianza": round(confianza, 2),
            "probabilidad_positiva": round(float(probabilidades[1]) * 100, 2),
            "probabilidad_negativa": round(float(probabilidades[0]) * 100, 2)
        })

    except Exception as e:
        print("Error de predicción:", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================================
# API PARA CONSULTAR METRICAS
# ==========================================

@app.route("/api/metricas", methods=["GET"])
def obtener_metricas():
    ruta = os.path.join(CARPETA_RESULTADOS, "metricas.json")

    if os.path.exists(ruta):
        return send_from_directory(CARPETA_RESULTADOS, "metricas.json")

    return jsonify({
        "error": "Todavía no existen métricas. Entrena el modelo."
    }), 404


# ==========================================
# MOSTRAR GRAFICAS (nombre que usa resultados.html)
# ==========================================

@app.route("/resultados/<path:archivo>")
def ver_resultado(archivo):
    return send_from_directory(CARPETA_RESULTADOS, archivo)


# ==========================================
# INICIAR SERVIDOR (siempre al final)
# ==========================================

if __name__ == "__main__":
    cargar_modelo()
    app.run(host="127.0.0.1", port=5000, debug=True)
