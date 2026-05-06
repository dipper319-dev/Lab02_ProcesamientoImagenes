"""FILTRO DE MEDIANA
Toma el valor central de los píxeles ordenados en la región.
Muy efectivo para eliminar ruido salt-and-pepper.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generar imagen de prueba
img = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Fondo blanco
img[50:150, 50:150] = [50, 50, 50]   # Cuadrado oscuro

# Añadir MUCHO ruido salt-and-pepper
img_ruidosa = img.copy()
num_sal = int(img_ruidosa.size * 0.025)  # 2.5% de píxeles blancos
num_pimienta = int(img_ruidosa.size * 0.025)  # 2.5% de píxeles negros

# Generar coordenadas aleatorias para sal (blanco)
for _ in range(num_sal):
    i, j, k = np.random.randint(0, 300), np.random.randint(0, 300), np.random.randint(0, 3)
    img_ruidosa[i, j, k] = 255

# Generar coordenadas aleatorias para pimienta (negro)
for _ in range(num_pimienta):
    i, j, k = np.random.randint(0, 300), np.random.randint(0, 300), np.random.randint(0, 3)
    img_ruidosa[i, j, k] = 0

# Aplicar filtro de mediana
# cv2.medianBlur(imagen, kernel_size)
resultado = cv2.medianBlur(img_ruidosa, 5)  # Kernel 5x5

# Mostrar resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))
axes[1].set_title('Con Ruido Salt-Pepper')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
axes[2].set_title('Filtro Mediana (5x5)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Filtro mediana aplicado exitosamente")