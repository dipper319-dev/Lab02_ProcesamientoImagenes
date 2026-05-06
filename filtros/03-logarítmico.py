"""FILTRO LOGARÍTMICO
Transformación: s = c * log(1 + r)
Comprime el rango dinámico, resalta detalles en zonas oscuras.
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

# Convertir a escala de grises para ver mejor el efecto
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convertir a float [0,1]
img_float = img_gray.astype(np.float32) / 255.0

# Aplicar transformación logarítmica: s = c * log(1 + r)
c = 100.0  # Constante de escala
img_log = c * np.log(1 + img_float)

# Normalizar resultado a [0,1]
img_log = img_log / np.max(img_log)

# Convertir de vuelta a [0,255]
resultado = (img_log * 255).astype(np.uint8)

# Mostrar resultados
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(resultado, cmap='gray')
axes[1].set_title(f'Filtro Logaritmico (c={c})')
axes[1].axis('off')

plt.suptitle('Filtro Logaritmico', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro logaritmico aplicado exitosamente")