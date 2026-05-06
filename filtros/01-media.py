"""FILTRO DE MEDIA
Promedia los píxeles en una región cuadrada del kernel.
Útil para suavizar imágenes y reducir ruido.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generar imagen de prueba (gradiente + ruido)
img = np.ones((300, 300, 3), dtype=np.uint8) * 127  # Fondo gris
img[50:150, 50:150] = [255, 0, 0]    # Cuadrado rojo (BGR)
img[150:250, 150:250] = [0, 255, 0]  # Cuadrado verde (BGR)
# Añadir ruido
ruido = np.random.normal(0, 15, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

# Aplicar filtro de media
# cv2.blur(imagen, (ancho_kernel, alto_kernel))
resultado = cv2.blur(img, (5, 5))  # Kernel 5x5

# Mostrar resultados
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original (con ruido)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
axes[1].set_title('Filtro Media (5x5)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("Filtro media aplicado exitosamente")