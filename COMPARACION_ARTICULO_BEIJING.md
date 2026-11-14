# Comparación configuración por configuración: Zhang & Zhang (2023) vs réplica Beijing

**Artículo:** Z. Zhang & S. Zhang (2023), *Modeling air quality PM2.5 forecasting using deep sparse attention-based transformer networks*, International Journal of Environmental Science and Technology, 20, 13535–13550. DOI: https://doi.org/10.1007/s13762-023-04900-1

| Elemento | Artículo | Réplica del proyecto | Dictamen |
|---|---|---|---|
| Datos | Beijing PM2.5 horario 2010–2014 | PRSA Beijing CSV 2010–2014 | Coincide en fuente/periodo esperado |
| Partición | Primeros 4 años train; 2014 test; 10% aleatorio del train para validación | 2010–2013 train; 2014 test; 10% cronológico final de train para validación | Parcial; la validación se cambió para evitar fuga temporal |
| Horizonte | h1 y horizontes 6/12/24/48 h | h1 | Coincide solo en h1 |
| Ventana | 24 horas | 24 horas | Coincide |
| Variables | 8 variables originales: PM2.5, dew point, temperature, pressure, wind direction, wind speed, snow hours, rain hours | Las 8 variables; `cbwd` one-hot genera 4 indicadores y el total queda en 11 columnas | Misma información base, representación distinta |
| Normalización | Todas las series a [0,1] | MinMax [0,1], parámetros ajustados solo con train | Coincide en rango; réplica evita ajustar con validación/test |
| Faltantes | Eliminación de outliers e interpolación lineal | Ffill causal hasta 24 h y mediana de train; imputación solo en entrada | Difiere deliberadamente; pérdida supervisada enmascarada |
| Arquitectura | STN de dos ramas; encoder sparse attention; decoder con masked sparse attention, atención multi-head cruzada estándar, FFN y proyección lineal | STN sparse attention con encoder en dos ramas, decoder masked sparse attention + FullAttention cruzada, FFN y proyección | Coincide en componentes generales |
| Ramas encoder | Una rama con un bloque sparse que recibe media secuencia; otra rama con dos bloques y destilación Conv1D k=3 + MaxPool stride 2 | `e_layers=3`; rama de secuencia completa con 3 bloques y rama de media secuencia con 2 bloques | Diferencia arquitectónica que debe declararse; la asignación/profundidad por rama no es idéntica a la descripción textual |
| Profundidad reportada | 3 encoders y 2 decoders | 3 capas en rama 1, 2 en rama 2; decoder de 2 capas | Decoder coincide; detalle de encoder no es equivalencia literal |
| `d_model`, cabezas, `d_ff` | No se especifican numéricamente en el texto disponible del artículo | `d_model=128`, `n_heads=8`, `d_ff=256` | No verificable contra el artículo; versión reducida por cómputo |
| Optimizador | Adam | Adam | Coincide |
| Learning rate | 1e-4 | 1e-4 para Beijing | Coincide |
| Batch size | 32 | 64 | Difiere |
| Épocas máximas | 200 | 40 | Difiere por límite de recursos |
| Pérdida | MSE | MSE enmascarada sobre PM2.5 observado | Adaptación necesaria para que los valores imputados no sean etiquetas |
| Framework | Texto del artículo menciona PyTorch/Sklearn y también TensorFlow para redes profundas | PyTorch + scikit-learn | PyTorch coincide con el marco citado; la mención a TensorFlow del artículo es inconsistente con su propia descripción |

## Lectura crítica

No se debe decir que la réplica es “idéntica al artículo”. Coinciden la ventana de 24 h, Adam, tasa inicial 1e-4, normalización al rango [0,1], la tarea h1 y los componentes principales de STN. Difieren batch, tope de épocas, política de validación, tratamiento de faltantes, codificación de la dirección del viento y la implementación detallada de las ramas del encoder. El artículo no permite verificar `d_model`, cabezas ni `d_ff`; por tanto, los valores de la réplica son decisiones computacionales del equipo, no coincidencias.

La comparación numérica de RMSE/MAE/R² debe volver a calcularse tras ejecutar el preentrenamiento con máscara, porque el preentrenamiento anterior incluía etiquetas imputadas.

## Referencia

Zhang, Z., & Zhang, S. (2023). Modeling air quality PM2.5 forecasting using deep sparse attention-based transformer networks. *International Journal of Environmental Science and Technology, 20*, 13535–13550. https://doi.org/10.1007/s13762-023-04900-1
