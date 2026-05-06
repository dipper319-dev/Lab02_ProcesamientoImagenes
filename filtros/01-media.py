"""FILTRO DE MEDIA
Promedia los píxeles en una región cuadrada del kernel.
Útil para suavizar imágenes y reducir ruido.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Ruta dinámica a imagen real ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(BASE_DIR, "dataset", "cat", "cats_00991.jpg")

# Cargar imagen real
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"No se encontró la imagen en: {IMG_PATH}")

# Añadir ruido para demostrar el efecto del filtro
ruido = np.random.normal(0, 15, img.shape).astype(np.int16)
img_ruidosa = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

# Aplicar filtro de media
# cv2.blur(imagen, (ancho_kernel, alto_kernel))
resultado = cv2.blur(img_ruidosa, (5, 5))  # Kernel 5x5

# Mostrar resultados
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))
axes[1].set_title('Con ruido')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
axes[2].set_title('Filtro Media (5x5)')
axes[2].axis('off')

plt.suptitle('Filtro de Media', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro media aplicado exitosamente")