"""FILTRO SOBEL
Detecta bordes calculando la primera derivada en dirección X e Y.
Permite detectar bordes horizontales, verticales o ambos.
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
img_suave = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Aplicar Sobel en X (bordes verticales)
# cv2.Sobel(imagen, ddepth, dx, dy, ksize)
sobel_x = cv2.Sobel(img_suave, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.uint8(np.absolute(sobel_x))

# Aplicar Sobel en Y (bordes horizontales)
sobel_y = cv2.Sobel(img_suave, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = np.uint8(np.absolute(sobel_y))

# Combinar X e Y (magnitud total de bordes)
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(img_gray, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(sobel_x, cmap='gray')
axes[0, 1].set_title('Sobel X (bordes verticales)')
axes[0, 1].axis('off')

axes[1, 0].imshow(sobel_y, cmap='gray')
axes[1, 0].set_title('Sobel Y (bordes horizontales)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sobel_xy, cmap='gray')
axes[1, 1].set_title('Sobel XY (combinado)')
axes[1, 1].axis('off')

plt.suptitle('Filtro Sobel', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro Sobel aplicado exitosamente")