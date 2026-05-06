"""FILTRO LOGARÍTMICO
Transformación: s = c * log(1 + r)
Comprime el rango dinámico, resalta detalles en zonas oscuras.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generar imagen con zonas claras y oscuras
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[0:100, :] = 20      # Zona oscura arriba
img[100:200, :] = 100   # Zona media
img[200:300, :] = 200   # Zona clara abajo
# Añadir patrón para ver el efecto (gradiente de 20 a 255)
for i in range(300):
    img[i, :] = int(np.clip(20 + (235 * i / 299), 0, 255))

# Convertir a float [0,1]
img_float = img.astype(np.float32) / 255.0

# Aplicar transformación logarítmica: s = c * log(1 + r)
c = 1.5  # Constante de escala
img_log = c * np.log(1 + img_float)

# Normalizar resultado a [0,1]
img_log = img_log / np.max(img_log)

# Convertir de vuelta a [0,255]
resultado = (img_log * 255).astype(np.uint8)
resultado_color = np.stack([resultado] * 3, axis=2)

# Mostrar resultados
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original (gradiente)')
axes[0].axis('off')

axes[1].imshow(resultado, cmap='gray')
axes[1].set_title(f'Filtro Logarítmico (c={c})')
axes[1].axis('off')

plt.tight_layout()
plt.show()

print("Filtro logarítmico aplicado exitosamente")