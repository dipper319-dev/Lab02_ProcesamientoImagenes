"""FILTRO CANNY
Detector de bordes que combina suavizado, gradientes y supresion de no maximos.
Produce bordes delgados y bien definidos con dos umbrales configurables.
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

# Convertir a grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Suavizar antes para reducir ruido
img_suave = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Aplicar Canny con distintos umbrales
# cv2.Canny(imagen, umbral_minimo, umbral_maximo)
canny_suave   = cv2.Canny(img_suave, 30,  100)  # umbrales bajos = más bordes
canny_medio   = cv2.Canny(img_suave, 80,  160)  # umbrales medios
canny_estricto = cv2.Canny(img_suave, 150, 250)  # umbrales altos = menos bordes

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(img_gray, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(canny_suave, cmap='gray')
axes[0, 1].set_title('Canny (umbrales 30-100)')
axes[0, 1].axis('off')

axes[1, 0].imshow(canny_medio, cmap='gray')
axes[1, 0].set_title('Canny (umbrales 80-160)')
axes[1, 0].axis('off')

axes[1, 1].imshow(canny_estricto, cmap='gray')
axes[1, 1].set_title('Canny (umbrales 150-250)')
axes[1, 1].axis('off')

plt.suptitle('Filtro Canny', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro Canny aplicado exitosamente")