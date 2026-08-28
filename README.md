# # 🚀 MLOps Pipeline: Sistema de Predicción y Monitoreo de Riesgo Crediticio

## 1. El Caso de Negocio
Las instituciones financieras enfrentan un desafío constante: maximizar la aprobación de créditos minimizando el riesgo de impago (default). En este proyecto, abordamos una base de datos donde más del 95% de los clientes son "Buenos Pagadores". 

El objetivo de negocio no es solo predecir con alta precisión quién pagará, sino **maximizar el Recall (Sensibilidad)** para detectar a tiempo a los clientes morosos (Clase 1), evitando pérdidas financieras. Además, debido a la naturaleza dinámica de la economía, el comportamiento y perfil de los aplicantes cambia con el tiempo. Por ello, el negocio requiere un sistema de **Machine Learning Operations (MLOps)** que no solo prediga, sino que monitoree la salud del modelo en producción para alertar sobre desviaciones en los datos (Data Drift).

---

## 2. El Proceso (De los Datos a Producción)

El desarrollo del pipeline se dividió en fases secuenciales:

*   **Ingeniería de Datos y Calidad:** Se eliminaron variables que generaban "Data Leakage" (fugas de información del futuro, como saldos en mora actuales) y se aplicaron filtros lógicos (edades entre 18 y 90 años, salarios mayores a 0).
*   **Feature Engineering:** Se crearon atributos derivados clave para el negocio, como la `carga_financiera` (Cuota Pactada / Salario Cliente), y se aplicaron transformaciones logarítmicas a variables con asimetría positiva extrema.
*   **Modelado y Ensamble (Avance 2):** Se evaluaron múltiples modelos (Regresión Logística, Random Forest, XGBoost, CatBoost y Stacking) bajo validación cruzada estratificada (5-Folds). El modelo campeón fue **LightGBM**, seleccionado por su equilibrio al capturar señales de mora en un entorno desbalanceado (usando `class_weight='balanced'`) y su extrema rapidez de inferencia.
*   **Monitoreo en Producción (Avance 3):** Se construyó una aplicación web interactiva utilizando **Streamlit** para la detección automatizada de Data Drift.

---

## 3. Arquitectura del Sistema de Monitoreo (Data Drift)

Para simular el entorno de producción y evaluar la estabilidad del modelo, el sistema divide los datos en dos segmentos:
1.  **Datos Históricos / Referencia:** Representan el 80% de los datos con los que el modelo fue entrenado en el pasado.
2.  **Datos Actuales / Producción:** Representan el 20% de los datos nuevos (simulando los clientes que solicitan créditos hoy).

El sistema de Streamlit somete estas dos poblaciones a rigurosas pruebas estadísticas de contraste de hipótesis:
*   **Prueba Kolmogorov-Smirnov (KS Test):** Aplicada a todas las variables numéricas (edad, salario, cantidad de créditos, etc.) para medir la distancia máxima entre sus distribuciones acumuladas.
*   **Prueba Chi-Cuadrado:** Aplicada a las variables categóricas (como el tipo laboral) para evaluar cambios significativos en la frecuencia y proporción de sus categorías.

**Regla de Negocio (Umbral):** Si el *P-Value* de cualquiera de las pruebas cae por debajo de **0.05**, se rechaza la hipótesis nula, confirmando estadísticamente un cambio en la distribución de la población (Drift).

---

## 4. Principales Hallazgos y Resultados del Monitoreo

Al ejecutar el motor de monitoreo sobre la población actual, el Dashboard emitió una alerta automática de estado **CRÍTICO 🔴**, fundamentada en los siguientes hallazgos:

1.  **Estabilidad Numérica:** Las 10 variables continuas y financieras del modelo (incluyendo `salario_cliente`, `cuota_pactada`, y `edad_cliente`) arrojaron un P-Value superior a 0.05 en el test de Kolmogorov-Smirnov. Esto indica que el poder adquisitivo y el perfil financiero general de los nuevos clientes se mantiene estable respecto al histórico.
2.  **Desviación Demográfica (El Data Drift):** El sistema detectó un cambio estructural orgánico en la variable categórica `tipo_laboral`. La prueba de Chi-Cuadrado arrojó un **P-Value de 0.0206**.
3.  **La Causa Técnica del Drift:** Al realizar la división de datos mediante un muestreo estratificado (`stratify=y`) para garantizar la misma proporción de morosos en el pasado y en el presente, la distribución de las profesiones quedó sujeta al azar. Esto simuló a la perfección un escenario real donde, de un mes a otro, el banco empieza a recibir aplicaciones de un nicho laboral diferente al habitual.

---

## 5. Recomendaciones de Negocio

La detección temprana de variaciones en la variable `tipo_laboral` es un indicador temprano de degradación del modelo predictivo (Model Decay). Si el LightGBM fue entrenado asumiendo ciertos riesgos para "Contratistas", pero la población actual está dominada por "Independientes", las predicciones comenzarán a fallar.

**Plan de Acción:**
1.  Pausar temporalmente las predicciones completamente automatizadas para la población afectada.
2.  Desviar los casos de los nuevos perfiles laborales a analistas de crédito humanos (Revisión Manual).
3.  Iniciar un ciclo de **Retraining** (reentrenamiento) del modelo LightGBM incluyendo el nuevo lote de datos de producción, permitiéndole aprender los patrones de riesgo de las nuevas distribuciones laborales.