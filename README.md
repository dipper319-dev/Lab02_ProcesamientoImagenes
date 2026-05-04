<div align="center">

# 🎓 Tarea 2 — Filtros de Imagen & Visión por Computador

> Procesamiento Digital de Imágenes · Fecha de entrega: **06 de mayo de 2026**

</div>

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [PARTE 1 — Filtros de Imagen](#-parte-1--filtros-de-imagen)
  - [1. Filtro de Media](#1-filtro-de-media)
  - [2. Filtro de Mediana](#2-filtro-de-mediana)
  - [3. Filtro Logarítmico](#3-filtro-logarítmico)
  - [4. Filtro de Cuadro Normalizado](#4-filtro-de-cuadro-normalizado-box-filter)
  - [5. Filtro Gaussiano](#5-filtro-gaussiano)
  - [6. Filtro Laplaciano](#6-filtro-laplaciano)
  - [7. Filtro Sobel](#7-filtro-sobel)
  - [8. Filtro Canny](#8-filtro-canny)


---

## 🧠 Descripción General

Este repositorio contiene la implementación completa de la **Tarea 2** del curso de Procesamiento Digital de Imágenes. El proyecto está dividido en dos partes:

- **Parte 1:** Investigación, documentación e implementación en Python + OpenCV de **8 filtros** de imagen clásicos, con su fórmula matemática, ejemplo numérico, ventajas y desventajas.
- **Parte 2:** Pipeline completo de clasificación de imágenes usando descriptores visuales (**HOG**, LBP, Haralick, SIFT), entrenamiento de clasificadores (**SVM / Red Neuronal**), evaluación con métricas y una **interfaz gráfica** funcional.

---



---

## 🔵 PARTE 1 — Filtros de Imagen

> Un **filtro de imagen** es una operación matemática que transforma el valor de cada píxel considerando su vecindario local, con el objetivo de modificar propiedades como nitidez, contraste o estructura de bordes.

---

### 1. Filtro de Media

**¿Qué hace?**
Reemplaza el valor de cada píxel con el **promedio aritmético** de todos los píxeles en su vecindario. Es el filtro de suavizado más simple. Al promediar los valores locales, atenúa variaciones bruscas de intensidad causadas por ruido aleatorio, produciendo una imagen más suave y homogénea. No distingue entre píxeles de ruido y bordes importantes, lo que produce pérdida de nitidez generalizada.

**Fórmula:**

$$g(x,y) = \frac{1}{M \cdot N} \sum_{(s,t) \in S_{xy}} f(s,t)$$

Donde $f(s,t)$ es el píxel original, $M \cdot N$ el total de píxeles en el vecindario (ej. 3×3 = 9) y $g(x,y)$ el píxel resultante.

**Ejemplo numérico (ventana 3×3):**

```
Vecindario:              Suma = 80+90+100+85+95+105+70+80+90 = 795
 [ 80   90  100 ]
 [ 85  [95] 105 ]        Media = 795 / 9 ≈ 88
 [ 70   80   90 ]
                         El píxel central (95) → se reemplaza por 88
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Simple y extremadamente rápido | Difumina bordes y detalles finos |
| Reduce ruido gaussiano efectivamente | Muy sensible a valores atípicos (outliers) |
| Fácil de escalar con el tamaño del kernel | Todos los píxeles tienen igual peso (poco natural) |
| Base para otros filtros más complejos | Puede generar artefactos en zonas de alta frecuencia |

---

### 2. Filtro de Mediana

**¿Qué hace?**
Es un filtro **no lineal** diseñado específicamente para eliminar ruido impulsivo (sal y pimienta). Ordena todos los píxeles del vecindario de menor a mayor y selecciona el valor central. Los píxeles con valores extremos (0 o 255) siempre quedan desplazados a los extremos del ordenamiento y nunca pueden ocupar la posición central, lo que los excluye automáticamente. Preserva los bordes mucho mejor que el filtro de media.

**Fórmula:**

$$g(x,y) = \text{mediana}\left\{f(s,t) \mid (s,t) \in S_{xy}\right\}$$

**Ejemplo numérico (ventana 3×3):**

```
Valores extraídos: [80, 90, 100, 85, 95, 105, 70, 80, 90]

Ordenados: [70, 80, 80, 85, →90←, 90, 95, 100, 105]
                              ↑
                        Posición central (5/9)

El píxel central (95) → se reemplaza por 90

Si hubiera ruido sal (255): el 255 queda en el extremo → no afecta la mediana ✅
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Excelente contra ruido sal y pimienta | Mayor costo computacional (requiere ordenamiento) |
| Preserva bordes mucho mejor que la media | Puede suavizar texturas finas o detalles pequeños |
| Inmune a valores atípicos extremos | Menos efectivo contra ruido gaussiano puro |
| No introduce valores nuevos en la imagen | Para ventanas grandes puede generar efecto "pintura" |

---

### 3. Filtro Logarítmico

**¿Qué hace?**
Es una **transformación de punto** (no considera vecindarios) que aplica una función logarítmica a cada píxel individualmente. Su objetivo es comprimir el rango dinámico: expande los valores oscuros (bajas intensidades) hacia rangos más altos y comprime los brillantes. Es especialmente útil en imágenes médicas, astronómicas o espectrales donde los rangos de intensidad son extremadamente amplios.

**Fórmula:**

$$g(x,y) = c \cdot \log\left(1 + f(x,y)\right) \qquad \text{donde} \quad c = \frac{255}{\log(1 + \max(f))}$$

Se suma 1 para evitar $\log(0)$. La constante $c$ normaliza el resultado al rango $[0, 255]$.

**Ejemplo numérico:**

```
Imagen 8 bits: max(f) = 255
c = 255 / log(256) = 255 / 2.408 ≈ 45.9

Píxel oscuro    f = 10  → g = 45.9 × log(11)  ≈  47.8
Píxel medio     f = 100 → g = 45.9 × log(101) ≈  92.0
Píxel brillante f = 200 → g = 45.9 × log(201) ≈ 105.7

El rango [10..200] se comprime a [48..106] en la salida
→ Las zonas oscuras ganan más espacio relativo ✅
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Mejora visibilidad en regiones oscuras | Puede saturar o perder detalle en zonas brillantes |
| Ideal para imágenes de alto rango dinámico | No elimina ruido; solo redistribuye contraste |
| Operación pixel a pixel: extremadamente rápida | La constante c requiere ajuste según la imagen |
| Útil en imágenes médicas y astronómicas | Introduce distorsión no lineal en la escala de grises |

---

### 4. Filtro de Cuadro Normalizado (Box Filter)

**¿Qué hace?**
Es la implementación del filtro de media expresada como **convolución con un kernel uniforme**. Todos los coeficientes son iguales a $1/(M \times N)$. Su ventaja clave sobre el filtro de media clásico es que puede implementarse con **imágenes integrales**, permitiendo calcular el filtro en tiempo O(1) por píxel independientemente del tamaño del kernel. OpenCV lo implementa con `cv2.boxFilter()`.

**Fórmula:**

$$g(x,y) = K * f(x,y) \qquad K = \frac{1}{M \cdot N}\begin{bmatrix}1 & 1 & \cdots & 1\\ \vdots & & & \vdots \\ 1 & 1 & \cdots & 1\end{bmatrix}$$

**Ejemplo (kernel 3×3 normalizado):**

```
       1   [ 1  1  1 ]
  K = ─── × [ 1  1  1 ]
       9   [ 1  1  1 ]

Aplicado sobre [80,90,100; 85,95,105; 70,80,90]:
g = (1/9)(80+90+100+85+95+105+70+80+90) = 795/9 ≈ 88

OpenCV: cv2.boxFilter(img, -1, (3,3), normalize=True)
        cv2.boxFilter(img, -1, (3,3), normalize=False)  # suma en vez de promedio
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Implementación O(1) con imágenes integrales | Borra bordes y detalles finos igual que la media |
| Escalable: el costo no aumenta con el kernel | Todos los píxeles tienen el mismo peso |
| Muy eficiente en aplicaciones en tiempo real | Poco efectivo contra ruido impulsivo |
| Base para algoritmos como SURF y tracking | Puede producir halos en bordes de alto contraste |

---

### 5. Filtro Gaussiano

**¿Qué hace?**
El filtro gaussiano es el **filtro de suavizado más utilizado** en visión por computador y la base de muchos algoritmos avanzados. Asigna pesos que siguen la forma de una campana de Gauss: los píxeles más cercanos al centro tienen mayor influencia y los pesos decrecen suavemente con la distancia. Es el único filtro lineal de suavizado completamente **isotrópico** (no tiene dirección preferida). El parámetro σ controla el nivel de suavizado.

**Fórmula:**

$$G(x,y) = \frac{1}{2\pi\sigma^2} \cdot e^{-\dfrac{x^2 + y^2}{2\sigma^2}}$$

**Ejemplo (kernel 3×3 aproximado con σ ≈ 1.0):**

```
       1   [ 1  2  1 ]
  K = ─── × [ 2  4  2 ]
      16   [ 1  2  1 ]

Pesos: centro = 4/16 = 0.25
       vecinos directos = 2/16 = 0.125
       esquinas = 1/16 = 0.0625

→ El píxel central pesa 4× más que las esquinas ✅

OpenCV: cv2.GaussianBlur(img, (5,5), sigmaX=1.0)
        # A mayor σ → más desenfoque
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Suavizado isotrópico: no tiene dirección preferida | Difumina bordes (aunque menos que la media) |
| Separable: reducción de complejidad computacional | σ debe elegirse cuidadosamente |
| Excelente para eliminar ruido gaussiano | Más costoso que el box filter |
| Base de Canny, SIFT, LoG y otros algoritmos clave | No efectivo contra ruido impulsivo de alta amplitud |

---

### 6. Filtro Laplaciano

**¿Qué hace?**
El filtro Laplaciano es un operador diferencial de **segundo orden** que amplifica las altas frecuencias (bordes, esquinas, detalles) de la imagen. Calcula la segunda derivada de la función de intensidad, lo que lo hace sensible a cualquier discontinuidad de intensidad sin importar su orientación: es un detector de bordes **omnidireccional**. Por ser tan sensible al ruido, casi siempre se usa después de un filtro gaussiano (combinación conocida como **LoG — Laplacian of Gaussian**).

**Fórmula:**

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} \approx f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4 \cdot f(x,y)$$

**Kernels discretos:**

```
Kernel estándar (4-conectado):    Kernel extendido (8-conectado):
  K1 = [  0   1   0 ]               K2 = [  1   1   1 ]
        [  1  -4   1 ]                     [  1  -8   1 ]
        [  0   1   0 ]                     [  1   1   1 ]

Ejemplo en fila: [100, 100, 100, 200, 200, 200]
En el borde (idx=3):
∇²f ≈ f(2) − 2·f(3) + f(4) = 100 − 400 + 200 = −100
→ Valor alto en módulo = BORDE DETECTADO ✅

OpenCV: cv2.Laplacian(img, cv2.CV_64F)
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Detecta bordes en todas las direcciones | Extremadamente sensible al ruido |
| Útil para realce de bordes (sharpening) | Produce bordes dobles (respuesta bipolar) |
| Identifica puntos de cruce por cero | No proporciona dirección del borde |
| Sencillo de implementar con kernels discretos | Requiere suavizado previo (LoG) para uso práctico |

---

### 7. Filtro Sobel

**¿Qué hace?**
El filtro Sobel calcula la **primera derivada** de la imagen en las direcciones horizontal y vertical usando kernels que combinan diferenciación y suavizado gaussiano. Esta combinación lo hace más robusto al ruido que el Laplaciano. Entrega tanto la **magnitud** del gradiente (qué tan fuerte es el borde) como su **dirección angular** (hacia dónde apunta el borde), información fundamental para algoritmos como Canny.

**Fórmulas:**

$$G_x = \begin{bmatrix}-1 & 0 & +1\\-2 & 0 & +2\\-1 & 0 & +1\end{bmatrix} \quad G_y = \begin{bmatrix}-1 & -2 & -1\\0 & 0 & 0\\+1 & +2 & +1\end{bmatrix}$$

$$G = \sqrt{G_x^2 + G_y^2} \qquad \theta = \arctan\!\left(\frac{G_y}{G_x}\right)$$

**Ejemplo:**

```
Parche 3×3:
  [  10   10   10  ]
  [  10   10  200  ]
  [  10   10  200  ]

Gx_centro ≈ (200+400+200) − (10+20+10) = 800 − 40 = 760
→ Gradiente horizontal muy alto → BORDE VERTICAL detectado ✅
→ θ nos indica la dirección perpendicular al borde

OpenCV: Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        G  = cv2.magnitude(Gx, Gy)
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Calcula magnitud Y dirección del gradiente | Produce bordes gruesos (más de 1 px de ancho) |
| Más robusto al ruido que el Laplaciano | Puede perder bordes diagonales de alta frecuencia |
| Kernels separables: eficiente computacionalmente | Respuesta no completamente isotrópica |
| Ampliamente soportado en todas las librerías | Sensible a ruido residual |

---

### 8. Filtro Canny

**¿Qué hace?**
El detector de bordes Canny (John Canny, 1986) es considerado el **detector de bordes óptimo** bajo tres criterios formales: buena detección (mínimos falsos positivos/negativos), buena localización (bordes detectados cerca de los reales) y respuesta única (un solo punto por borde real). Combina cuatro etapas de procesamiento para producir bordes delgados, limpios y bien conectados. Es el estándar de la industria para detección de bordes.

**Algoritmo (4 etapas):**

$$\text{1. Suavizado: } I_s = G_\sigma * I \quad\rightarrow\quad \text{2. Gradiente Sobel: } G, \theta \quad\rightarrow\quad \text{3. NMS} \quad\rightarrow\quad \text{4. Histéresis}(T_l, T_h)$$

**Proceso completo:**

```
ETAPA 1 — Suavizado Gaussiano (σ típico: 1.0 – 1.4):
  I_s = GaussianBlur(I, σ)    ← elimina ruido antes de derivar

ETAPA 2 — Gradiente Sobel:
  G = √(Gx² + Gy²)    θ = arctan(Gy/Gx)

ETAPA 3 — Non-Maximum Suppression:
  Para cada píxel: si G(x,y) NO es máximo local en dirección θ → G(x,y) = 0
  → Adelgaza bordes gruesos a bordes de 1 píxel de ancho ✅

ETAPA 4 — Umbralización con Histéresis (Th=200, Tl=100):
  G > 200          → Borde FUERTE  → aceptado definitivamente
  100 < G < 200    → Borde DÉBIL   → solo si conecta con un borde fuerte
  G < 100          → Descartado

OpenCV: cv2.Canny(img, threshold1=100, threshold2=200)
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Bordes delgados de exactamente 1 píxel | Más lento que Sobel y Laplaciano |
| Robusto al ruido por el suavizado previo | Requiere ajuste de dos umbrales (Tl, Th) |
| Conecta bordes fragmentados con histéresis | Puede perder bordes débiles legítimos con Tl alto |
| Óptimo bajo criterios matemáticos formales | El suavizado gaussiano puede borrar bordes muy finos |

---

