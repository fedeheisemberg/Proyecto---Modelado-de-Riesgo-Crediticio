# Credit Scoring App - README

## 📌 Descripción del Proyecto

Esta aplicación de Streamlit implementa un sistema de credit scoring que predice la probabilidad de mora de un cliente basado en características como ingreso, nivel de deuda, edad y nivel de estudios. El proyecto incluye:

- Análisis exploratorio de datos
- Modelado con Árboles de Decisión y Random Forest
- Evaluación de modelos con métricas de desempeño
- Interfaz interactiva para predecir el riesgo de nuevos clientes

## 🚀 Características Principales

- **Visualización interactiva** de datos y modelos
- **Sistema de scoring** que asigna puntajes de 0 a 1000
- **Interpretación clara** de resultados con recomendaciones
- **Comparación de modelos** con métricas detalladas
- **Dashboard completo** con múltiples secciones

## 📊 Estructura de la Aplicación

La aplicación tiene 5 secciones principales:

1. **Inicio**: Descripción del proyecto y vista previa de los datos
2. **Exploración de Datos**: Análisis visual de las variables
3. **Modelado**: Entrenamiento y visualización de modelos
4. **Evaluación**: Métricas de desempeño y curvas ROC
5. **Predicción**: Interfaz para predecir riesgo de nuevos clientes

## 🛠️ Requisitos Técnicos

- Python 3.7+
- Librerías principales:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## ⚙️ Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/credit-scoring-app.git
   cd credit-scoring-app
   ```

2. Crear y activar un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```

## 📂 Estructura de Archivos

```
credit-scoring-app/
├── app.py                # Código principal de la aplicación
├── score_credit.csv      # Dataset de ejemplo (debe ser proporcionado)
├── README.md             # Este archivo
├── requirements.txt      # Dependencias del proyecto
└── images/               # (Opcional) Capturas de pantalla
```

## 📈 Dataset

El proyecto requiere un archivo `score_credit.csv` con las siguientes columnas:
- `Ingreso`: Ingreso mensual del cliente (numérico)
- `Nivel_Deuda`: Porcentaje de deuda respecto al ingreso (numérico, 0-100)
- `Edad`: Edad del cliente (entero)
- `Nivel_Estudios`: Nivel educativo (categórico)
- `Mora`: Variable objetivo (0 = No mora, 1 = Mora)


## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

---

**Nota**: Asegúrate de incluir el archivo `score_credit.csv` en el mismo directorio que la aplicación o modificar la ruta en el código para apuntar a tu dataset.