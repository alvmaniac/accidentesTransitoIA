# Predicción de Severidad en Accidentes de Tránsito con Machine Learning

## 1. Introducción

Este aplicativo predice la severidad de accidentes de tránsito a partir de características del incidente aplicando técnicas de inteligencia artificial. Utiliza algoritmos de aprendizaje automático, técnicas de balanceo de clases y visualización de datos para comprender mejor los factores asociados a los distintos niveles de severidad.

---

## 2. Marco Teórico – Tecnologías y Librerías Utilizadas

### 2.1 Aprendizaje Automático (Machine Learning)

El aprendizaje automático es una rama de la inteligencia artificial que permite a los sistemas aprender y mejorar automáticamente a partir de los datos. En este proyecto se empleó aprendizaje supervisado, utilizando etiquetas históricas de severidad de accidentes para predecir nuevas ocurrencias.

El algoritmo central es **Random Forest**, un método de ensamble que combina múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste. Cada árbol es entrenado con una muestra aleatoria del dataset y vota por la clase final.

- **Python**: Lenguaje principal de desarrollo.
---

### 2.2 Desbalance de Clases y SMOTE

El desbalance de clases ocurre cuando algunas categorías (como los accidentes fatales) están poco representadas en comparación con otras (como los accidentes leves). Esto puede sesgar los modelos hacia la clase mayoritaria.

Para enfrentar este problema se utilizó **SMOTE (Synthetic Minority Over-sampling Technique)**, una técnica que crea instancias sintéticas de la clase minoritaria generando nuevos puntos entre observaciones cercanas del mismo grupo. Esto mejora la capacidad del modelo para reconocer clases infrecuentes.

- **Imbalanced-learn**: Aplicación de técnicas como SMOTE para balancear clases.
---

### 2.3 Bibliotecas de Procesamiento de Datos

- **Pandas**: Permite la manipulación eficiente de estructuras tabulares, como la carga, limpieza y análisis de datasets.
- **NumPy**: Biblioteca para el cálculo numérico avanzado con arrays multidimensionales.

---

### 2.4 Visualización de Datos

- **Matplotlib** y **Seaborn** son bibliotecas de visualización usadas para explorar gráficamente el dataset. Permiten crear:
  - Histogramas de distribución de clases.
  - Gráficos de barras para frecuencia de variables.
  - Mapas de calor (heatmaps) de la matriz de confusión.

Estas herramientas visuales ayudan a interpretar el comportamiento del modelo y la calidad de las predicciones.

---

### 2.5 Evaluación del Modelo

Para evaluar el desempeño del modelo se utilizaron las siguientes métricas:

- **Scikit-learn**: Para modelado, métricas de evaluación y preprocesamiento.
- **Accuracy (Precisión global)**: Proporción de predicciones correctas sobre el total.
- **Precision (Precisión por clase)**: Exactitud de las predicciones positivas.
- **Recall (Exhaustividad o sensibilidad)**: Capacidad del modelo para encontrar todos los casos positivos.
- **F1-score**: Media armónica entre precisión y recall.
- **Matriz de confusión**: Representación tabular de verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos por clase.

Estas métricas son especialmente importantes cuando las clases están desbalanceadas.

---

### 2.6 Flask: Aplicación Web para Predicción

**Flask** es un microframework para construir aplicaciones web en Python. En este proyecto se utilizó para desarrollar una **Aplicaciòn Web**, la cual:

- Recibe datos de entrada de repositorios CSV y Base de datos.
- Procesa los datos y los pasa al modelo entrenado.
- Presenta la predicción al usuario final en una aplicación Web.
- Crea una nueva predicciòn en base a datos ingresados por el usuario

---

## 3. Descripción del Dataset

El conjunto de datos contiene registros de accidentes de tránsito, provienen de dos tipos que son cvs y base de datos con las siguientes columnas relevantes:

- `Reference_Number`
- `Accident_Date`
- `Time_24hr`
- `1st_Road_Class`
- `Road_Surface`
- `Lighting_Conditions`
- `Weather_Conditions`
- `Casualty_Class`
- `Sex_of_Casualty`
- `Type_of_Vehicle`
- `Did_Police_Officer_Attend_Scene_of_Accident`
- `Casualty_Severity` (variable objetivo, clases en la variable:** `Fatal`, `Serious`, `Slight`.)

**Total de registros CSV:** ~800 

![image](https://github.com/user-attachments/assets/9fad8993-6ce4-4af5-b08b-f6298e8cd916)

**Total de registros base de datos:** ~551

![image](https://github.com/user-attachments/assets/c8789cc6-c61b-400c-8a38-9a79023acec0)

---

## 4. Pasos Realizados en el Proyecto

1. **Carga y exploración inicial del dataset. (`data_loader.py`)**
2. **Limpieza y procesamiento de datos (`preprocess.py`):**
   - Eliminación de valores nulos o mal codificados (`NA`).
   - Codificación de variables categóricas.
   - División de datos 'train_test_split' (80% entrenamiento, 20% prueba).
5. **Entrenamiento y evaluación del moelo (`model.py`):**
   - Aplicación de SMOTE para balancear las clases minoritarias
   - RandomForestClassifier con `n_estimators=100`.
   - Métricas: precisión, recall, F1-score, accuracy.
   - Matriz de confusión.
6. **Visualización de Graficos e indicadores (`visualizations.py`):**
   - Creación de grafico histogramas.
   - Creación de mapa con localidad de accidentes.
7. **Exposición como aplicación Web(`app.py`):**
   - Artefacto principal del cual se hace el llamado a todo el proceso para crear la presentación web en un `dashboard.html` con indicadores, graficos y   mapas.

8. Estructura del Proyecto

![image](https://github.com/user-attachments/assets/6d0d9720-b8a4-45fb-af17-504b3a164496)

### 4.1 Visualizaciones generadas

- **Histograma de severidad de accidentes:** para ver distribución de clases.


![image](https://github.com/user-attachments/assets/f5c71e86-2e10-4aee-bb93-b72405266c99)


- **Matriz de confusión:** visualización con `seaborn.heatmap()`.


![image](https://github.com/user-attachments/assets/609e5972-93e8-401e-835b-21b1abac137a)


- **Gráficos de importancia de características:** para interpretar el modelo.


![image](https://github.com/user-attachments/assets/3bec2cc4-44ba-4517-836b-ac498d5cbded)


- **Realizar una nueva predicción:**.

![image](https://github.com/user-attachments/assets/f61a473f-5667-4338-9a64-57ce91f672bd)

![image](https://github.com/user-attachments/assets/6ae09018-c543-47ca-bf2a-6d5758c7b121)

---

## 5. Conclusiones

- El modelo logró un **accuracy general del 71%**, pero mostró un **desempeño pobre en la clase 'Fatal'**, con F1-score = 0.00.
- El desbalance de clases fue un reto significativo; la aplicación de SMOTE ayudó parcialmente.
- Se recomienda explorar modelos más robustos frente al desbalance (como `BalancedRandomForestClassifier` o `XGBoost`) y buscar más datos de clases críticas.
- La integración con Flask permite una solución reproducible y lista para producción.

---

## 6. Bibliografía

- Scikit-learn Documentation: https://scikit-learn.org/
- Imbalanced-learn: https://imbalanced-learn.org/
- MLflow: https://mlflow.org/

---


