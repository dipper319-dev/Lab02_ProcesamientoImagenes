"""FILTRO LAPLACIANO
Detecta bordes calculando la segunda derivada de la imagen.
Resalta zonas de cambio brusco de intensidad en todas direcciones.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ── Ruta dinámica a imagen RANDOM del dataset ───────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER   = os.path.join(BASE_DIR, "dataset", "cat")
IMG_PATH = os.path.join(FOLDER, random.choice(os.listdir(FOLDER)))

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"No se encontró la imagen en: {IMG_PATH}")

print(f"Imagen cargada: {IMG_PATH}")

# Convertir a grises (Laplaciano trabaja mejor en grises)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizar antes para reducir ruido
img_suave = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Aplicar filtro Laplaciano
# cv2.Laplacian(imagen, ddepth, ksize)
laplaciano = cv2.Laplacian(img_suave, cv2.CV_64F, ksize=3)

# Convertir a uint8 para visualizar
laplaciano_abs = np.uint8(np.absolute(laplaciano))

# Laplaciano realzado: suma con original para resaltar bordes
realzado = cv2.addWeighted(img_gray, 1.0, laplaciano_abs, 0.8, 0)

# Mostrar resultados
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(laplaciano_abs, cmap='gray')
axes[1].set_title('Laplaciano (bordes)')
axes[1].axis('off')

axes[2].imshow(realzado, cmap='gray')
axes[2].set_title('Original + Laplaciano (realzado)')
axes[2].axis('off')

plt.suptitle('Filtro Laplaciano', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro laplaciano aplicado exitosamente")