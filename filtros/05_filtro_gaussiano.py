"""FILTRO GAUSSIANO
Suaviza la imagen usando una función gaussiana como kernel.
Reduce ruido preservando mejor los bordes que el filtro de media.
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

# Añadir ruido gaussiano para demostrar el efecto
ruido = np.random.normal(0, 25, img.shape).astype(np.int16)
img_ruidosa = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

# Aplicar filtro gaussiano con distintos kernels
# cv2.GaussianBlur(imagen, (ancho, alto), sigmaX)
resultado_3x3 = cv2.GaussianBlur(img_ruidosa, (3, 3), 0)
resultado_5x5 = cv2.GaussianBlur(img_ruidosa, (5, 5), 0)
resultado_9x9 = cv2.GaussianBlur(img_ruidosa, (9, 9), 0)

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Con ruido')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(resultado_3x3, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Gaussiano (3x3)')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(resultado_5x5, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Gaussiano (5x5)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(resultado_9x9, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Gaussiano (9x9)')
axes[1, 1].axis('off')

plt.suptitle('Filtro Gaussiano', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro gaussiano aplicado exitosamente")