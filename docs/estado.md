# Qué es esta web y en qué punto está

> Documento de trabajo del repositorio. Las reglas duras y el índice están
> en [CLAUDE.md](../CLAUDE.md). **Si descubres un fallo nuevo, añádelo aquí
> en la misma sesión.**


Este repositorio empezó como notebooks de ML y se está convirtiendo (notebooks de ML) en una web educativa interactiva que clona la estética de mlu-explain.github.io. **La web ya está publicada en https://aleetreny.github.io/ATLAS/**.

## Arquitectura
- El sitio vive en la **raíz del repo** (`index.html` + `assets/` + una carpeta por artículo), no en `docs/`. GitHub Pages está configurado a main → `/(root)`, con `.nojekyll`.
- **Sin build step**: HTML artesanal + ES modules + D3 v7 + KaTeX vendorizados en `assets/js/vendor/`. Nada de frameworks ni npm.
- Scrollytelling con `IntersectionObserver` propio en `assets/js/scrolly.js`. Dispara por **banda central** (`rootMargin: -45% 0px -45% 0px`), no por umbral de área.
- Helpers compartidos de gráficos en `assets/js/chart.js` (`makeChart`, `drawAxes`, `drawGrid`, `axisLabels`, `tooltip`, `spread`).
- Servir en local: `.claude/launch.json` define "atlas-web" (python http.server 8737 sobre la raíz del repo).

## Estado: módulos 1.1 y 1.2 completos (salvo KDA), módulo 2.1 abierto. Doce artículos
| # | Artículo | Carpeta | Tema |
|---|---|---|---|
| 1 | Classical Regression (OLS, polinomial, GLM, PLS) | `classical-regression/` | `#ff9900` |
| 2 | Regularization (Ridge, Lasso, ElasticNet) | `regularization/` | `#7c5aed` |
| 3 | Robust Regression (Huber, Theil-Sen, RANSAC) | `robust-regression/` | `#007faa` |
| 4 | Non-Parametric (KNN, kernels, procesos gaussianos) | `non-parametric-regression/` | `#0e8a55` |
| 5 | Gradient Boosting (+ XGBoost/LightGBM/CatBoost) | `gradient-boosting/` | `#df2a5d` |
| 6 | Deep Learning for Tables (TabNet, NAM, EBM) | `deep-learning-tables/` | `#3b4cca` |
| 7 | Probabilistic Classification (logística, naive Bayes) | `probabilistic-classification/` | `#a8410f` |
| 8 | Geometric Classification (SVM, kernels, LDA/QDA) | `geometric-classification/` | `#4d7c0f` |
| 9 | Trees and Ensembles (árbol, bagging, RF, extratrees) | `trees-and-ensembles/` | `#86198f` |
| 10 | Imbalanced Learning (accuracy trap, SMOTE, ADASYN) | `imbalanced-learning/` | `#0f766e` |
| 11 | Calibration (Platt, isotónica/PAV, ECE, Brier) | `calibration/` | `#9f1239` |
| 12 | K-Means (Lloyd, k-means++, silhouette, fallos) | `kmeans/` | `#a16207` |
| 13 | Gaussian Mixtures (EM vivo, arrastrar medias, BIC) | `gmm/` | `#155e75` |
| 14 | DBSCAN · HDBSCAN (eps/minPts vivos, k-dist, trampa) | `dbscan/` | `#6d28d9` |
| 15 | Hierarchical + BIRCH (dendro con cuchilla, Lance-Williams) | `hierarchical/` | `#b91c1c` |
| 16 | SVR + Kernel Ridge (tubo eps, KRR vivo, identidad GP) | `svr-kernel-ridge/` | `#475569` |
| 17 | Kernel Discriminants (dial del techo lineal 0.878, KDA propio) | `kda/` | `#0c4a6e` |
| 18 | Features Before Learning (HOG vivo, SIFT/ORB medidos) | `classical-features/` | `#78350f` |
| 19 | Viola-Jones (imagen integral, Haar, cascada viva) | `viola-jones/` | `#1e40af` |
| 20 | Convolutional Networks (editor de kernel, filtros aprendidos, traslación) | `lenet/` | `#15803d` |
| 21 | Depth (degradación medida, gradiente que explota, atajos residuales) | `resnet/` | `#4c1d95` |
| 22 | After Depth (presupuesto, escalera ConvNeXt, suelo de ruido dibujado) | `convnext/` | `#b45309` |
| 23 | Where (IoU arrastrable, factura de ventanas, detector vivo, NMS) | `detection/` | `#0891b2` |

Los seis pasaron una **revisión completa** (commit `5e4852b`, 2026-07-25): auditoría estática en paralelo más un barrido en navegador de cada control, cada paso de scrolly y cada etiqueta a 1425px y 375px. 73 hallazgos aplicados. Todas las páginas tienen ahora `canonical` + Open Graph, los 20 sliders tienen `aria-label`, y cada cierre enlaza al siguiente artículo.

Dos de esos hallazgos eran errores numéricos publicados: `polyFit` del artículo 1 formaba `X'X` (grado 9 daba r² test 0.681 en vez de 0.649, invirtiendo la moraleja) y el lasso del artículo 2 no convergía por debajo de λ=1e-3. Ver [la receta de un artículo](receta-de-articulo.md).

**Módulos 1.1 y 1.2 completos; 1.3 (visión) con los artículos 18-23 publicados. El 21 cumplió la promesa del 20 (ReLU gana a tanh con profundidad: 89,37/83,77 a 20 capas) y desmintió midiendo el tópico del gradiente que se desvanece (crece 1.271×). El 23 cerró el arco que abrió el 19: un clasificador de ventanas sobre un recorte y el mismo en una pasada convolucional dan el mismo número exacto. Siguiente en 1.3: segmentación (U-Net/Mask R-CNN/SAM, sin generador todavía) y NeRF. También pendientes: módulo 2.2 (PCA, que el cierre de hierarchical ya anuncia) y los chips restantes de 2.1. Ver atlas-inflight-2126.**

**Artículo 23** (`detection/`) es el primero que ejecuta un modelo entrenado ENTERO en el navegador: 42.814 pesos plegados y cuantizados a float16 en base64 (111 kB), convoluciones en JavaScript sobre los píxeles de un PNG, contrastados contra referencias que el generador calculó en float64 con los pesos ya cuantizados. El patrón del contrato de export (`w_offset`/`w_size`/`b_offset`/`b_size`, `pool_after`, `order`, `decode`, `score`) es reutilizable para cualquier artículo futuro que quiera hacer lo mismo, y `detection/js/detkit.js` es el forward-pass genérico. Ojo con `halfToFloat`: una versión ingenua aplasta los subnormales a cero.

**Estructura de la portada (revisión integral 2026-07-27)**: cada módulo tiene su propia galería de tarjetas (antes las 4 de clustering vivían en la del módulo 1) y el orden es taxonómico, no cronológico. Las cifras de las frases numéricas del artículo 21 las compone `resnet/js/prose.js` desde el JSON al cargar, con ramas condicionales por si una re-tirada mueve el resultado; el patrón vale para artículos futuros. La cadena de cierres está completa: cada artículo desemboca en el siguiente y los puentes entre módulos son kda→classical-features y calibration→kmeans.

## Módulo 1.3, infraestructura de visión (nueva)
- **`assets/js/imagery.js`**: el segundo camino de render del sitio (canvas, no SVG). `loadImage`, `imageToGray` (luma 601, igual que rgb2gray), `makeCanvas` (dpr, max-width), `paintGray` (zoom entero vía bitmap offscreen porque `putImageData` ignora transformaciones), `pixelGrid`, `divergingColour`, `canvasLabel`. Requiere http: `getImageData` se contamina en `file://`.
- **Imágenes**: PNG sin pérdida, y el generador mide sobre los **valores de 8 bits cuantizados** que el navegador leerá, no sobre el float original, o la comprobación en carga nunca cuadra.
- **Datasets pesados fuera del repo**: `~/.atlas_vision_data` (MNIST vía torchvision la primera vez, luego IDX crudo con numpy). `src/utils/vision_data.py` lo encapsula. Instalado en la máquina: scikit-image 0.26, torchvision 0.28 (torch 2.13 CPU, ~6 ms/paso en CNN pequeña). No hay opencv. `fetch_openml` de MNIST devolvió 504; CIFAR-10 tarda mucho en bajar.
- **HOG**: replicable exactamente (gradiente de diferencias centradas con bordes a cero, orientaciones mod 180, celda = suma de magnitudes / área). Coincide con skimage a 1e-7 si se usa su epsilon (1e-5) en la normalización L1.
- **Haar**: skimage **resta** el primer rectángulo (valor = -A+B-C...). Con el signo al revés el error era 1.3e+02.
- **Trampa de umbrales discretos**: con pocos stumps la puntuación toma poquísimos valores distintos; un umbral guardado como "alcanzable menos 1e-9" pierde la epsilon al redondear a 6 decimales en JSON y mueve de lado todas las ventanas que caen justo ahí (1.377 de 10.816). Usar siempre **punto medio entre valores alcanzables**.
- **La precisión de exportación ES lo que mide el guardia**: con pesos a 5 decimales y mapas de referencia a 4, el navegador cuadraba a 7.1e-5 y parecía un error de implementación; subiendo a 7 y 6 decimales cuadra a 9.7e-7, que es exactamente el redondeo. Si el guardia da un número feo, mirar primero cuántos decimales se escribieron.
- **`toLocaleString()` sin locale** escribe "25.600" en una máquina española, que en inglés se lee 25,6. Todo el repo pasa `'en-US'`; es fácil olvidarlo en un archivo nuevo.

**Artículo 20** (LeNet) salió con **dos resultados en contra del guion previsto, y el artículo se reescribió alrededor de ellos**: (1) la traslación no es invariancia: la CNN aguanta 1-3 px mejor que el MLP (0.943 vs 0.789 a 1 px) pero **las dos se hunden por debajo del azar a 4 px** (0.0976), y a 5-6 px las curvas se cruzan; la invariancia la da la aumentación (0.4942 vs 0.0976 a 4 px), no la arquitectura. (2) **tanh GANA a ReLU** en esta red (0.9758 vs 0.9698, y llega antes a la meta de pérdida: época 4 contra 6), porque con 2 capas convolucionales y Adam no hay saturación que arreglar; el artículo lo publica como está y explica el régimen. Números anclados: LeNet 61.706 params vs MLP 62.020 (78 ocultas), 0.9698 vs 0.9350 (151 errores de 5.000 contra 325, 53,5% eliminados); conv1 = 156 params donde una capa densa del mismo tamaño necesitaría 3.692.640; de los 6 filtros aprendidos **solo UNO suma ~0** (detector de bordes de verdad), los otros llegan a +2.01. Generador: 33 s, entrena 6 redes.

**Artículos 16-17** (los chips que faltaban): el 16 reutiliza los coches y el split exacto del artículo 4 (semilla 7); su kernel ridge se resuelve EN VIVO en el navegador (Cholesky 294×294 por tick) y la identidad KRR≡media GP exige el MISMO prior de media cero (centrar la y del GP la rompe: gap 1.17 → 7.6e-13 al no centrar). El 17 implementa kernel Fisher en numpy (sklearn no lo trae); su rejilla de gammas se eligió por sondeo para contar el arco entero (0.01 clava el techo lineal 0.878 al decimal; 1000 memoriza: 1.000/0.711); el techo lineal de las lunas es un barrido de 721 ángulos, y el dial del lector guarda su récord personal contra él.

**Artículos 13-15** comparten datasets con el 12 (blobs semilla 19, galería de fallos, vino) a propósito: el arco anillos es -0.003 (km) → 0.002 (gmm) → 1.000 (dbscan). Lecciones nuevas medidas: en los blobs cizallados **tied gana a full por BIC** (comparten la cizalla); en el vino 13-D **diag gana a full** (314 parámetros sobreajustan 178 filas); el demo de PAV y la curva de tamaño necesitan scores CONTINUOS y muy descalibrados; el **ARI con ruido-como-clase premia borrar un cluster entero** (0.973 llamando ruido al difuso: por eso el widget de la trampa sigue dos metas y no el ARI); el threshold de BIRCH en 10-D debe ser ~3.5 o no resume nada (con 1.7 tardó 17 min con 90k subclusters); y las coordenadas se **redondean antes de medir** o el guardia navegador-vs-sklearn dispara por pares en el filo del radio. Generadores: `generate_gmm_data.py`, `generate_dbscan_data.py`, `generate_hierarchical_data.py` (rápidos, sin covtype).

**Artículos 10 y 11** comparten dataset: covtype (581.012 celdas) con la clase 4 (cottonwood/willow, 2.747 celdas) y la mayoría adelgazada a prevalencia 1% (274.700 celdas, semilla 17, split 60/40; el 11 corta además el train en fit/calibración 75/25). El 10 mide que las seis intervenciones de rebalanceo EMPEORAN el AUPRC (0,723 → 0,60-0,67) e inflan p media a ~4% en un mundo al 1%; el 11 lo repara (Platt/isotónica, cruce de la curva de tamaño en 10k celdas) y cierra con la descomposición de Brier. El 12 usa blobs semilla 19 (inercia de referencia 501,7) y el vino del artículo 1 (ARI 0,897, 172/178). Regenerar: `generate_imbalanced_data.py`, `generate_calibration_data.py`, `generate_kmeans_data.py` (los dos primeros tardan ~20 min por el fit del SMOTE logistic).

**Artículo 7** mide logística contra naive Bayes sobre 569 biopsias de Wisconsin: 4,7 puntos de exactitud de diferencia y un factor DIEZ en log loss (0,070 contra 0,718). El control es QDA, que es naive Bayes con las covarianzas puestas. La prueba está en el diagrama de fiabilidad: naive Bayes metió 108 de 171 pacientes en el cubo más bajo prometiendo 0,00024, y siete de ellos tenían cáncer. Borrar las columnas duplicadas NO lo arregla (Brier empeora de 0,062 a 0,086). Regenerar: `src/utils/generate_classification_data.py`.

**Artículo 9** mide la ecuación de ESL, ρσ² + (1−ρ)σ²/B, con ρ y σ² leídos de los árboles y la curva comprobada contra ellos: muestrear columnas divide ρ por 2,9 (0,053 → 0,018) y el suelo por 2,7, y a 200 árboles la fórmula predice 0,001218 contra 0,001215 medido. **Ojo con qué es ρ**: NO es la correlación entre dos árboles de un bosque sobre datos fijos (ahí son condicionalmente independientes, la varianza cae como 1/B hasta cero y no hay suelo), sino entre árboles a través de conjuntos de entrenamiento independientes. Medirlo mal hace que la fórmula falle por un 99%. También: la importancia por impureza da 5,7× más peso a una columna de ruido continuo que a una binaria, y un árbol no extrapola en absoluto. Regenerar: `src/utils/generate_trees_data.py`.

**Artículo 8** mide el margen (0,383 con dos vectores de soporte; un solo punto añadido lo estrecha 153 veces), el truco del kernel como transición continua, el cruce LDA/QDA a 30 filas por clase con 13 features (por debajo de 14, QDA ni se ajusta), y el dato que casi nunca se publica: sobre 30 features reales el kernel gana 0,36 puntos, menos que el ruido entre pliegues. Regenerar: `src/utils/generate_geometric_data.py`.

## Librerías instaladas (todas las cifras del sitio son ejecutadas, no citadas)
xgboost 3.2.0, lightgbm 4.7.0, catboost 1.2.10, torch 2.13.0+cpu, pytorch-tabnet, interpret 0.7.8, scikit-learn 1.8.0, Python 3.11.7, Windows AMD64.

Ojo con sklearn 1.8: el primer parámetro posicional de `MLPRegressor` **ya no es** `hidden_layer_sizes` sino `loss`, así que hay que pasarlo por nombre o falla con un error que no lo parece.

**Artículo 5** publica un benchmark de las cinco implementaciones de boosting fijadas a la misma capacidad (300 árboles, profundidad 3, lr 0.1), porque comparar cada librería con sus defaults mide quién los tiene más atrevidos. Caben en 3,3 $ de RMSE mientras xgboost va 51 veces más rápido que sklearn exacto. Regenerar: `src/utils/generate_boosting_data.py`.

**Artículo 6** mide siete modelos sobre los mismos diamantes: gradient boosting 526,2 $ en 0,7s gana; el mejor neuronal (MLP) 660,2 $ en 13,9s; TabNet 700,1 $ con una desviación entre semillas de ±100,5 que es su verdadero titular. El resultado interesante es el **EBM con términos por pares: 587,8 $**, a un 12% del ganador siendo un modelo que se lee entero. Incluye el test de rotación de Grinsztajn sobre datos reales (boosting +17,8%, mlp −6,8%, ridge −0,1%) con la corrección que casi nadie cuenta: boosting rotado sigue ganando a la red sin rotar. Regenerar: `src/utils/generate_deeptabular_data.py`.

Datasets y generadores en `src/utils/generate_*.py`; cada uno escribe el JSON del artículo e incluye valores de referencia de scikit-learn que el navegador comprueba al cargar.

Ver [la receta de un artículo](receta-de-articulo.md) para el formato de cada artículo y las lecciones ya pagadas, y [el flujo de despliegue](despliegue.md) para el flujo de despliegue.
