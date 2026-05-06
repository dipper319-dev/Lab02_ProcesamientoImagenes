"""FILTRO DE CUADRO NORMALIZADO (Box Filter)
Filtro rectangular que promedia píxeles en una región rectangular.
Útil para difuminado direccional (horizontal, vertical o ambos).
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

# Aplicar box filter con distintos kernels
# cv2.boxFilter(imagen, ddepth, (ancho, alto), normalize=True)
resultado_5x5 = cv2.boxFilter(img, -1, (5, 5),  normalize=True)
resultado_3x7 = cv2.boxFilter(img, -1, (3, 7),  normalize=True)  # efecto vertical
resultado_7x3 = cv2.boxFilter(img, -1, (7, 3),  normalize=True)  # efecto horizontal

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(resultado_5x5, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Box Filter (5x5)')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(resultado_3x7, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Box Filter (3x7 - efecto vertical)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(resultado_7x3, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Box Filter (7x3 - efecto horizontal)')
axes[1, 1].axis('off')

plt.suptitle('Filtro de Cuadro Normalizado', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro de cuadro normalizado aplicado exitosamente")