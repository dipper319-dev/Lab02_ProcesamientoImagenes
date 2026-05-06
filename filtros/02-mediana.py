"""FILTRO DE MEDIANA
Toma el valor central de los píxeles ordenados en la región.
Muy efectivo para eliminar ruido salt-and-pepper.
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

# Añadir ruido salt-and-pepper
img_ruidosa = img.copy()
num_sal      = int(img_ruidosa.size * 0.025)
num_pimienta = int(img_ruidosa.size * 0.025)

for _ in range(num_sal):
    i, j, k = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1]), np.random.randint(0, 3)
    img_ruidosa[i, j, k] = 255

for _ in range(num_pimienta):
    i, j, k = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1]), np.random.randint(0, 3)
    img_ruidosa[i, j, k] = 0

# Aplicar filtro de mediana
# cv2.medianBlur(imagen, kernel_size)
resultado = cv2.medianBlur(img_ruidosa, 5)  # Kernel 5x5

# Mostrar resultados
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))
axes[1].set_title('Con Ruido Salt-Pepper')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
axes[2].set_title('Filtro Mediana (5x5)')
axes[2].axis('off')

plt.suptitle('Filtro de Mediana', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Filtro mediana aplicado exitosamente")