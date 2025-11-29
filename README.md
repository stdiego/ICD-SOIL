# 🌱 ICD Soil  
### *La calidad del dato para cultivar mejor*

---

## 📌 Descripción general

**ICD Soil** es una plataforma analítica que utiliza datos abiertos e inteligencia artificial para evaluar, visualizar y mejorar la calidad de la información disponible sobre suelos agrícolas en Colombia. Su principal objetivo es apoyar decisiones técnicas, productivas y ambientales mediante el cálculo del **Índice de Calidad del Dato (ICD)** para cada variable de fertilidad del suelo.

El proyecto se desarrolla en el marco del **Concurso Datos al Ecosistema 2025** y trabaja con la base oficial:  
📍 *Resultados de Análisis de Laboratorio de Suelos en Colombia* (datos.gov.co)

---

## 🎯 Propósito

El análisis de suelos es fundamental para definir planes de fertilización, mejorar el rendimiento agrícola y proteger los recursos naturales. Sin embargo, la información recopilada presenta desafíos como registros incompletos, valores extremos, inconsistencias y baja continuidad temporal.

**ICD Soil transforma esos datos dispersos en conocimiento confiable**, detectando anomalías, midiendo confiabilidad estadística, creando modelos predictivos y permitiendo visualizar la calidad de los datos a nivel nacional, regional y municipal.

---

## 🧪 ¿Qué hace ICD Soil?

| Componente | Descripción |
|-----------|-------------|
| 🧹 **Limpieza y normalización de datos** | Depuración, estandarización y validación de registros. |
| 🔍 **Detección de anomalías** | Identificación de valores atípicos mediante Isolation Forest y métricas estadísticas. |
| 🧠 **Modelos predictivos ML** | Modelos de regresión multivariable entrenados por variable para evaluar coherencia. |
| 📈 **Forecasting temporal** | Pronósticos por variable utilizando métodos de series de tiempo. |
| 🧮 **Cálculo del ICD por variable y región** | Escala 0–1 que refleja confiabilidad del dato. |
| 📊 **Visualizaciones interactivas** | Ranking, mapas, distribuciones, tendencias y simuladores. |
| 💾 **Exportación y API local** | Descarga de métricas e indicadores para uso externo. |

---

## 🧱 Arquitectura del proyecto


---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|----------|--------------|
| Lenguaje | Python 3.10+ |
| Data Processing | pandas, numpy, pyarrow |
| Machine Learning | scikit-learn |
| Detección de anomalías | Isolation Forest |
| Series temporales | statsmodels / Holt-Winters |
| Visualización | seaborn, plotly, geopandas |
| Interfaz | Streamlit |
| Infraestructura de datos | datos.gov.co API |

---

## 🚀 ¿Cómo ejecutar?

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/<usuario>/ICD-Soil.git
cd ICD-Soil
