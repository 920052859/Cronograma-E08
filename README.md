# 🚇 Predicción del Cronograma de Trabajo con IA · Estación E4-8

Programa de **machine learning** que predice el estado del cronograma de cada
actividad del proyecto **Diseño, Procura y Construcción de la Infraestructura
Civil de la Estación E4-8 de la Línea 4 del Metro de Lima y Callao** durante
**enero y febrero de 2026**.

Cada actividad se clasifica en: 🟢 **A tiempo** · 🟡 **Con riesgo de retraso**
· 🔴 **Retrasada**.

> Todo el programa está en **un solo archivo** (`prediccion_cronograma_e08.py`)
> para que el despliegue sea a prueba de errores: no hay carpetas ni imports
> internos que puedan romperse al subir a GitHub.

---

## ✨ Qué hace

- Entrena un clasificador **Random Forest** multiclase (`scikit-learn`).
- Predice, para **cada actividad**, su estado en enero y febrero de 2026.
- Genera la **tabla de resultados** con: estado actual, predicción enero 2026,
  predicción febrero 2026, probabilidad de retraso y nivel de riesgo estimado.
- Entrega el **análisis para el Jefe de Proyecto**: qué actividades requieren
  atención inmediata para evitar impactos en el cronograma general.
- Exporta la tabla a **CSV y Excel**.

## 🧮 Actividades reales

Las **32 actividades** provienen de la distribución mensual del presupuesto del
proyecto (partidas con avance programado en enero/febrero 2026): 12 de
Ingeniería, 8 de Permisos y 12 de Gestión. El avance y el desempeño se simulan
de forma **reproducible**, ya que no se cuenta con el registro de producción en
vivo; para uso real, estas variables se alimentan de Microsoft Project y los
reportes de avance.

## 🧠 Variables del modelo

Avance planificado, avance real, **SPI** (Schedule Performance Index =
avance real / avance planificado), rendimiento, restricciones pendientes,
disponibilidad de recursos, dependencias pendientes, holgura, criticidad,
retrasos previos y riesgo de la fase.

La predicción de **febrero** incorpora un **arrastre de riesgo**: si una
actividad viene con riesgo o retraso en enero, sus condiciones de febrero se
degradan, reflejando la propagación de desviaciones a las actividades
sucesoras.

## 🚀 Ejecución local

```bash
pip install -r requirements.txt
streamlit run prediccion_cronograma_e08.py
```

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube **estos dos archivos a la raíz** de un repositorio de GitHub:
   `prediccion_cronograma_e08.py` y `requirements.txt`.
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Selecciona el repositorio, la rama `main` y como *Main file path*
   escribe `prediccion_cronograma_e08.py`.
4. **Deploy**.

## ⚖️ Alcance

El modelo es una herramienta de **apoyo a la decisión**: prioriza y anticipa,
pero la decisión sobre reprogramaciones y asignación de recursos corresponde al
Jefe de Proyecto y su equipo.

---

*Ejercicio · Predicción de cronograma con IA · Maestría en Project Management*
