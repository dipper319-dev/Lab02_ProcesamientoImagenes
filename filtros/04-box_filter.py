"""BOX FILTER
Filtro rectangular que promedia píxeles en una región rectangular.
Útil para difuminado direccional (horizontal, vertical o ambos).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generar imagen de prueba con bordes marcados
img = np.ones((300, 300, 3), dtype=np.uint8) * 200
img[75:225, 75:225] = [0, 0, 0]  # Cuadrado negro
# Añadir líneas para ver el efecto del filtro
img[150, :] = [255, 255, 255]  # Línea horizontal blanca
img[:, 150] = [255, 255, 255]  # Línea vertical blanca

# Aplicar box filter
# cv2.boxFilter(imagen, ddepth, (ancho, alto), normalize=True)
resultado_5x5 = cv2.boxFilter(img, -1, (5, 5), normalize=True)
resultado_3x7 = cv2.boxFilter(img, -1, (3, 7), normalize=True)  # Vertical
resultado_7x3 = cv2.boxFilter(img, -1, (7, 3), normalize=True)  # Horizontal

# Mostrar resultados
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(resultado_5x5, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Box Filter (5x5)')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.cvtColor(resultado_3x7, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Box Filter (3x7 - Vertical)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(resultado_7x3, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Box Filter (7x3 - Horizontal)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print("Box filter aplicado exitosamente")