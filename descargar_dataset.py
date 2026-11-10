
import kagglehub
import shutil
import os

print("DESCARGANDO DATASET IMDB...")

# Descargar dataset original
path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

print("Dataset descargado en:", path)

# Crear carpeta dataset
os.makedirs("dataset", exist_ok=True)

# Buscar archivo CSV
archivo = os.path.join(path, "IMDB Dataset.csv")

# Copiar archivo al proyecto
if os.path.exists(archivo):

    shutil.copy2(
        archivo,
        "dataset/IMDB Dataset.csv"
    )

    print("Dataset copiado correctamente.")

else:
    print("No se encontró el archivo CSV.")

print("PROCESO FINALIZADO")
