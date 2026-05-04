import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern
from pathlib import Path

# ── Rutas dinámicas ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_procesado")
OUTPUT_DIR  = os.path.join(BASE_DIR, "modelos")

clases = ["cat", "dog", "other"]
etiquetas_map = {"cat": 0, "dog": 1, "other": 2}

# ── Parámetros LBP ──────────────────────────────────────
LBP_RADIO  = 1
LBP_PUNTOS = 8 * LBP_RADIO

# ── Funciones ───────────────────────────────────────────
def extraer_hog(img):
    caracteristicas, _ = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    return caracteristicas

def extraer_lbp(img):
    lbp = local_binary_pattern(img, LBP_PUNTOS, LBP_RADIO, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_PUNTOS + 3), density=True)
    return hist

# ── Extracción ──────────────────────────────────────────
X_hog, X_lbp, y = [], [], []

for clase in clases:
    carpeta = os.path.join(DATASET_DIR, clase)
    label   = etiquetas_map[clase]

    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        img  = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        X_hog.append(extraer_hog(img))
        X_lbp.append(extraer_lbp(img))
        y.append(label)

X_hog = np.array(X_hog)
X_lbp = np.array(X_lbp)
y     = np.array(y)

# ── Guardar ─────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_hog.npy"), X_hog)
np.save(os.path.join(OUTPUT_DIR, "X_lbp.npy"), X_lbp)
np.save(os.path.join(OUTPUT_DIR, "y.npy"),     y)

print("✅ Características extraídas!")
print(f"   HOG shape : {X_hog.shape}")
print(f"   LBP shape : {X_lbp.shape}")
print(f"   Etiquetas : {y.shape}  (0=gato, 1=perro, 2=otro)")