import cv2
import os
import numpy as np
from pathlib import Path

# ── Rutas dinámicas ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR  = os.path.join(BASE_DIR, "dataset_procesado")
IMG_SIZE    = (128, 128)

clases = ["cat", "dog", "other"]

# ── Crear carpetas de salida ────────────────────────────
for clase in clases:
    Path(f"{OUTPUT_DIR}/{clase}").mkdir(parents=True, exist_ok=True)

# ── Procesar imágenes ───────────────────────────────────
resumen = {}

for clase in clases:
    input_path  = os.path.join(DATASET_DIR, clase)
    output_path = os.path.join(OUTPUT_DIR, clase)
    imagenes    = os.listdir(input_path)
    contador    = 0

    for nombre in imagenes:
        ruta = os.path.join(input_path, nombre)
        img  = cv2.imread(ruta)

        if img is None:
            continue

        # 1. Resize a 128x128
        img = cv2.resize(img, IMG_SIZE)

        # 2. Convertir a escala de grises
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Filtro Gaussiano (suavizar ruido)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        cv2.imwrite(os.path.join(output_path, nombre), img_gray)
        contador += 1

    resumen[clase] = contador

# ── Resultado ───────────────────────────────────────────
print("✅ Preprocesamiento completado!")
print(f"   🐱 Gatos procesados : {resumen['cat']}")
print(f"   🐶 Perros procesados: {resumen['dog']}")
print(f"   🌍 Otros procesados : {resumen['other']}")
print(f"   📁 Guardado en     : {OUTPUT_DIR}")