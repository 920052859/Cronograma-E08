"""
=============================================================================
PREDICCIÓN DEL CRONOGRAMA DE TRABAJO CON INTELIGENCIA ARTIFICIAL
Diseño, Procura y Construcción de la Infraestructura Civil de la
Estación E4-8 · Línea 4 del Metro de Lima y Callao (proyecto EPC)
=============================================================================

Programa de machine learning que predice el estado del cronograma de cada
actividad durante ENERO 2026 y FEBRERO 2026, clasificándola en:

    • A tiempo
    • Con riesgo de retraso
    • Retrasada

Las actividades son REALES: se extrajeron de la distribución mensual del
presupuesto del proyecto (partidas con avance programado en enero/febrero
2026). El estado de avance y desempeño se simula de forma reproducible, ya
que no se cuenta con el registro de producción en vivo.

Todo el programa está en un solo archivo para simplificar el despliegue.
Ejecutar con:  streamlit run prediccion_cronograma_e08.py
=============================================================================
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Cronograma E4-8 · IA",
    page_icon="🚇",
    layout="wide",
)

CLASES = ["A tiempo", "Con riesgo de retraso", "Retrasada"]
COLOR_CLASE = {
    "A tiempo": "#2E9E5B",
    "Con riesgo de retraso": "#E8A200",
    "Retrasada": "#C8102E",
}
COLOR_RIESGO = {"Bajo": "#2E9E5B", "Medio": "#E8A200", "Alto": "#C8102E"}

# Riesgo base por fase (las fases con dependencias externas son más volátiles)
RIESGO_FASE = {
    "Ingeniería": 0.30,   # depende de revisiones internas y del concedente
    "Permisos": 0.55,     # depende de entidades reguladoras externas
    "Gestión": 0.18,      # actividades internas más controlables
    "Construcción": 0.45,
    "Procura": 0.38,
    "Otro": 0.35,
}

# ---------------------------------------------------------------------------
# ACTIVIDADES REALES DEL PROYECTO (programadas para enero/febrero 2026)
# Extraídas de la hoja "Detalle de Costo" del presupuesto del proyecto.
# ---------------------------------------------------------------------------
ACTIVIDADES = [
    {"codigo": "1.1.1.1", "descripcion": "Definición de alcance técnico y criterios generales de diseño", "fase": "Ingeniería", "costo": 18000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "1.1.1.2", "descripcion": "Elaboración del expediente de diseño civil de la Estación E08", "fase": "Ingeniería", "costo": 62000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.1.3", "descripcion": "Revisión técnica y levantamiento de observaciones del expediente", "fase": "Ingeniería", "costo": 25600, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.2.1", "descripcion": "Planos de estructuras principales de caja de estación", "fase": "Ingeniería", "costo": 36000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.2.2", "descripcion": "Planos de pantallas, pilas pilote y elementos de contención", "fase": "Ingeniería", "costo": 28000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.2.3", "descripcion": "Planos de losas estructurales de cubierta, vestíbulo y fondo", "fase": "Ingeniería", "costo": 32000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.2.4", "descripcion": "Planos de accesos, drenaje, pozo de bombeo y estructuras secundarias", "fase": "Ingeniería", "costo": 20000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.2.5", "descripcion": "Emisión IFC y control documental de planos para construcción", "fase": "Ingeniería", "costo": 10000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.3.1", "descripcion": "Memoria de cálculo de caja de estación y losas principales", "fase": "Ingeniería", "costo": 30000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.3.2", "descripcion": "Memoria de cálculo de pantallas, pilas pilote y contención", "fase": "Ingeniería", "costo": 34000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.3.3", "descripcion": "Criterios de diseño estructural, geotécnico y constructivo", "fase": "Ingeniería", "costo": 16000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "1.1.3.4", "descripcion": "Revisión de consistencia entre memorias, planos y especificaciones", "fase": "Ingeniería", "costo": 15000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.1.1", "descripcion": "Licencia de edificación obtenida", "fase": "Permisos", "costo": 13333, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.1.2", "descripcion": "Autorización de ocupación de vía obtenida", "fase": "Permisos", "costo": 8000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.1.3", "descripcion": "Permiso para cierre de vías obtenido", "fase": "Permisos", "costo": 6667, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.1.4", "descripcion": "Permiso para desvío vehicular obtenido", "fase": "Permisos", "costo": 8667, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.2.1", "descripcion": "Aprobación de expediente obtenida", "fase": "Permisos", "costo": 12000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.2.2", "descripcion": "Autorización de inicio de obras obtenida", "fase": "Permisos", "costo": 8000, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.2.3", "descripcion": "Conformidad técnica obtenida", "fase": "Permisos", "costo": 6667, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "3.1.2.4", "descripcion": "Certificados regulatorios obtenidos", "fase": "Permisos", "costo": 6667, "activo_enero": 0, "activo_febrero": 1},
    {"codigo": "5.1.1.1", "descripcion": "Acta de constitución aprobada", "fase": "Gestión", "costo": 12000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.1.1.2", "descripcion": "Sponsor designado", "fase": "Gestión", "costo": 8000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.1.2.1", "descripcion": "Registro de interesados completado", "fase": "Gestión", "costo": 7000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.1.2.2", "descripcion": "Matriz poder-interés elaborada", "fase": "Gestión", "costo": 8000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.2.1.1", "descripcion": "Plan de gestión aprobado", "fase": "Gestión", "costo": 27000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.2.1.2", "descripcion": "Línea base integrada aprobada", "fase": "Gestión", "costo": 18000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.2.2.1", "descripcion": "EDT aprobada", "fase": "Gestión", "costo": 17000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.2.2.2", "descripcion": "Diccionario EDT completado", "fase": "Gestión", "costo": 13000, "activo_enero": 1, "activo_febrero": 0},
    {"codigo": "5.2.3.1", "descripcion": "Cronograma base aprobado", "fase": "Gestión", "costo": 24000, "activo_enero": 1, "activo_febrero": 1},
    {"codigo": "5.2.3.2", "descripcion": "Ruta crítica definida", "fase": "Gestión", "costo": 14000, "activo_enero": 1, "activo_febrero": 1},
    {"codigo": "5.2.4.1", "descripcion": "Presupuesto base aprobado", "fase": "Gestión", "costo": 20000, "activo_enero": 1, "activo_febrero": 1},
    {"codigo": "5.2.4.2", "descripcion": "Línea base de costos establecida", "fase": "Gestión", "costo": 12000, "activo_enero": 1, "activo_febrero": 1},
]

FEATURES = [
    "avance_planificado", "avance_real", "spi", "rendimiento",
    "restricciones_pendientes", "disponibilidad_recursos",
    "dependencias_pendientes", "holgura_dias", "es_critica",
    "retrasos_previos", "riesgo_fase",
]


# ---------------------------------------------------------------------------
# 1. ETIQUETADO: regla de negocio que define el estado del cronograma
# ---------------------------------------------------------------------------
def indice_atraso(f):
    """Índice continuo de atraso a partir de las variables de control.

    Combina el SPI (Schedule Performance Index = avance real / avance
    planificado) con penalizaciones por restricciones, dependencias,
    bajo rendimiento, falta de recursos y criticidad.
    """
    idx = (1 - f["spi"]) * 1.1                         # desviación de avance
    idx += (1 - f["rendimiento"]) * 0.35
    idx += (1 - f["disponibilidad_recursos"]) * 0.30
    idx += f["restricciones_pendientes"] * 0.045
    idx += f["dependencias_pendientes"] * 0.055
    idx += f["retrasos_previos"] * 0.04
    idx += f["es_critica"] * 0.04
    idx += f["riesgo_fase"] * 0.12
    idx += max(0, (3 - f["holgura_dias"])) * 0.02      # holgura baja penaliza
    return idx


# Umbrales calibrados sobre la distribución del índice (~45% a tiempo,
# ~27% con riesgo, ~28% retrasada).
UMBRAL_A_TIEMPO = 0.47
UMBRAL_RIESGO = 0.64


def clasificar(idx):
    if idx < UMBRAL_A_TIEMPO:
        return "A tiempo"
    if idx < UMBRAL_RIESGO:
        return "Con riesgo de retraso"
    return "Retrasada"


def nivel_riesgo(prob_retrasada, prob_riesgo):
    """Nivel de riesgo estimado a partir de las probabilidades del modelo."""
    if prob_retrasada >= 0.50:
        return "Alto"
    if (prob_retrasada + prob_riesgo) >= 0.60:
        return "Medio"
    return "Bajo"


# ---------------------------------------------------------------------------
# 2. DATOS DE ENTRENAMIENTO (histórico sintético reproducible)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_entrenamiento(n=1500, seed=42):
    """Genera el histórico sintético usando la MISMA función de snapshot que
    las actividades reales, de modo que la distribución de entrenamiento
    coincida con la de aplicación (consistencia train/predicción).
    """
    rng = np.random.default_rng(seed)
    fases = list(RIESGO_FASE.keys())
    meses = ["actual", "enero", "febrero"]
    filas = []
    for _ in range(n):
        fase = fases[rng.integers(0, len(fases))]
        mes = meses[rng.integers(0, 3)]
        arr = float(rng.choice([0.0, 0.0, 0.12, 0.25]))
        f = snapshot_actividad({"fase": fase}, mes, rng, arrastre=arr)
        idx = indice_atraso(f) + rng.normal(0, 0.04)   # ruido
        f["estado"] = clasificar(idx)
        filas.append(f)
    return pd.DataFrame(filas)


@st.cache_resource(show_spinner=False)
def entrenar(_df):
    X = _df[FEATURES]
    y = _df["estado"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=9, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    modelo.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, modelo.predict(X_te))
    rep = classification_report(y_te, modelo.predict(X_te),
                                output_dict=True, zero_division=0)
    return modelo, acc, rep


# ---------------------------------------------------------------------------
# 3. SNAPSHOTS DE LAS ACTIVIDADES REALES (estado actual, enero, febrero)
# ---------------------------------------------------------------------------
def snapshot_actividad(act, mes, rng, arrastre=0.0):
    """Genera el vector de variables de una actividad en un mes dado.

    'arrastre' propaga el riesgo del mes anterior (febrero hereda las
    desviaciones de enero: si una actividad o sus predecesoras venían mal,
    febrero empeora).
    """
    rf = RIESGO_FASE[act["fase"]]
    # avance planificado esperado según el mes de ejecución
    if mes == "actual":
        avance_plan = float(np.clip(rng.normal(35, 12), 5, 70))
    elif mes == "enero":
        avance_plan = float(np.clip(rng.normal(55, 15), 10, 95))
    else:  # febrero
        avance_plan = float(np.clip(rng.normal(80, 15), 20, 100))

    # SPI base influido por el riesgo de fase y el arrastre
    spi = float(np.clip(rng.normal(1.03 - rf * 0.22 - arrastre, 0.10), 0.5, 1.2))
    avance_real = float(np.clip(avance_plan * spi, 0, 100))
    restricciones = int(np.clip(rng.poisson(0.6 + rf * 1.8 + arrastre * 3), 0, 8))
    recursos = float(np.clip(rng.normal(0.92 - rf * 0.22 - arrastre * 0.3, 0.08), 0.3, 1.0))
    dependencias = int(np.clip(rng.poisson(0.5 + rf * 1.2 + arrastre * 2), 0, 6))
    return {
        "avance_planificado": round(avance_plan, 1),
        "avance_real": round(avance_real, 1),
        "spi": round(spi, 3),
        "rendimiento": round(float(np.clip(rng.normal(0.90 - rf * 0.18, 0.08), 0.4, 1.0)), 3),
        "restricciones_pendientes": restricciones,
        "disponibilidad_recursos": round(recursos, 3),
        "dependencias_pendientes": dependencias,
        "holgura_dias": round(float(np.clip(rng.exponential(3.5), 0, 20)), 1),
        "es_critica": int(act["fase"] in ("Ingeniería", "Permisos") or rng.random() < 0.3),
        "retrasos_previos": int(np.clip(rng.poisson(0.4 + rf * 0.8), 0, 5)),
        "riesgo_fase": rf,
    }


@st.cache_data(show_spinner=False)
def construir_resultados(_modelo_id):
    """Construye la tabla de resultados aplicando el modelo a cada actividad.

    _modelo_id es un parámetro ficticio para la caché; el modelo se toma del
    estado global (ya entrenado y determinista).
    """
    rng = np.random.default_rng(123)
    filas = []
    for act in ACTIVIDADES:
        # Estado actual
        s_act = snapshot_actividad(act, "actual", rng)
        estado_actual = clasificar(indice_atraso(s_act))

        # Predicción enero (solo si la actividad está activa en enero)
        s_ene = snapshot_actividad(act, "enero", rng)
        p_ene = MODELO.predict_proba(pd.DataFrame([s_ene])[FEATURES])[0]
        clase_ene = MODELO.classes_[np.argmax(p_ene)]
        prob = dict(zip(MODELO.classes_, p_ene))
        pred_ene = clase_ene if act["activo_enero"] else "—"

        # Arrastre de riesgo hacia febrero
        arr = 0.0
        if clase_ene == "Retrasada":
            arr = 0.25
        elif clase_ene == "Con riesgo de retraso":
            arr = 0.12

        s_feb = snapshot_actividad(act, "febrero", rng, arrastre=arr)
        p_feb = MODELO.predict_proba(pd.DataFrame([s_feb])[FEATURES])[0]
        clase_feb = MODELO.classes_[np.argmax(p_feb)]
        prob_feb = dict(zip(MODELO.classes_, p_feb))
        pred_feb = clase_feb if act["activo_febrero"] else "—"

        # Nivel de riesgo estimado (peor caso entre los meses activos)
        probs_activos = []
        if act["activo_enero"]:
            probs_activos.append(prob)
        if act["activo_febrero"]:
            probs_activos.append(prob_feb)
        pr = max(p.get("Retrasada", 0) for p in probs_activos) if probs_activos else 0
        prg = max(p.get("Con riesgo de retraso", 0) for p in probs_activos) if probs_activos else 0
        riesgo = nivel_riesgo(pr, prg)

        filas.append({
            "Código": act["codigo"],
            "Actividad": act["descripcion"],
            "Fase": act["fase"],
            "Costo (USD)": act["costo"],
            "Estado actual": estado_actual,
            "Predicción enero 2026": pred_ene,
            "Predicción febrero 2026": pred_feb,
            "Prob. retraso (%)": round(pr * 100),
            "Nivel de riesgo": riesgo,
        })
    return pd.DataFrame(filas)


# ===========================================================================
# EJECUCIÓN
# ===========================================================================
df_train = generar_entrenamiento()
MODELO, ACC, REPORTE = entrenar(df_train)
df_res = construir_resultados(_modelo_id=1)

# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.title("🚇 Predicción del Cronograma de Trabajo con IA")
st.markdown(
    "**Estación E4-8 · Línea 4 del Metro de Lima y Callao** — "
    "Diseño, Procura y Construcción de la Infraestructura Civil (EPC)"
)
st.caption(
    "Modelo de machine learning que predice el estado del cronograma de cada "
    "actividad en enero y febrero de 2026: a tiempo, con riesgo de retraso o "
    "retrasada."
)

# KPIs generales
c1, c2, c3, c4 = st.columns(4)
c1.metric("Actividades analizadas", len(df_res))
c2.metric("🔴 Riesgo alto", int((df_res["Nivel de riesgo"] == "Alto").sum()))
c3.metric("🟡 Riesgo medio", int((df_res["Nivel de riesgo"] == "Medio").sum()))
c4.metric("Exactitud del modelo", f"{ACC*100:.0f}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tabla de resultados",
    "🎯 Análisis para el Jefe de Proyecto",
    "📈 Visualización",
    "🧠 Metodología y modelo",
])

# ---------------------------------------------------------------------------
# TAB 1 · TABLA DE RESULTADOS
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Estado actual y predicción por actividad")
    st.caption(
        "Para cada actividad se muestra su estado actual, la predicción para "
        "enero y febrero de 2026 y el nivel de riesgo estimado. El guion (—) "
        "indica que la actividad no tiene avance programado ese mes."
    )

    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        fases_sel = st.multiselect("Filtrar por fase", sorted(df_res["Fase"].unique()),
                                   default=sorted(df_res["Fase"].unique()))
    with fcol2:
        riesgo_sel = st.multiselect("Filtrar por nivel de riesgo", ["Alto", "Medio", "Bajo"],
                                    default=["Alto", "Medio", "Bajo"])

    df_view = df_res[df_res["Fase"].isin(fases_sel) & df_res["Nivel de riesgo"].isin(riesgo_sel)]

    def color_estado(val):
        if val in COLOR_CLASE:
            return f"background-color: {COLOR_CLASE[val]}; color: white; font-weight:600"
        return ""

    def color_riesgo(val):
        if val in COLOR_RIESGO:
            return f"background-color: {COLOR_RIESGO[val]}; color: white; font-weight:600"
        return ""

    styled = (
        df_view.style
        .map(color_estado, subset=["Estado actual", "Predicción enero 2026", "Predicción febrero 2026"])
        .map(color_riesgo, subset=["Nivel de riesgo"])
        .format({"Costo (USD)": "${:,.0f}", "Prob. retraso (%)": "{:.0f}%"})
    )
    st.dataframe(styled, use_container_width=True, height=560, hide_index=True)

    # Descargas
    csv = df_res.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_res.to_excel(writer, index=False, sheet_name="Resultados")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Descargar tabla (CSV)", csv,
                       "resultados_cronograma_E08.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️ Descargar tabla (Excel)", buffer.getvalue(),
                       "resultados_cronograma_E08.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 · ANÁLISIS PARA EL JEFE DE PROYECTO
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Actividades que requieren mayor atención")

    criticas = df_res[df_res["Nivel de riesgo"] == "Alto"].sort_values("Prob. retraso (%)", ascending=False)
    medias = df_res[df_res["Nivel de riesgo"] == "Medio"].sort_values("Prob. retraso (%)", ascending=False)

    st.markdown(
        f"El modelo identifica **{len(criticas)} actividades de riesgo alto** y "
        f"**{len(medias)} de riesgo medio** entre las {len(df_res)} programadas "
        "para enero y febrero de 2026. Estas son las que el Jefe de Proyecto "
        "debe priorizar para evitar impactos en el cronograma general."
    )

    if len(criticas):
        st.markdown("#### 🔴 Prioridad alta — intervención inmediata")
        for _, r in criticas.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{r['Código']} · {r['Actividad']}** ({r['Fase']})  \n"
                    f"Probabilidad de retraso: **{r['Prob. retraso (%)']}%** · "
                    f"Enero: *{r['Predicción enero 2026']}* · Febrero: *{r['Predicción febrero 2026']}*"
                )

    if len(medias):
        st.markdown("#### 🟡 Prioridad media — seguimiento cercano")
        st.dataframe(
            medias[["Código", "Actividad", "Fase", "Predicción enero 2026",
                    "Predicción febrero 2026", "Prob. retraso (%)"]],
            use_container_width=True, hide_index=True,
        )

    # Recomendaciones por fase
    st.divider()
    st.markdown("#### Lectura por fase")
    resumen_fase = (
        df_res.groupby("Fase")["Prob. retraso (%)"].mean().round(0)
        .reset_index().sort_values("Prob. retraso (%)", ascending=False)
    )
    for _, r in resumen_fase.iterrows():
        fase = r["Fase"]
        if fase == "Permisos":
            nota = ("Dependen de entidades reguladoras externas; conviene "
                    "gestionar anticipadamente licencias y autorizaciones para "
                    "no bloquear el inicio de obras.")
        elif fase == "Ingeniería":
            nota = ("El expediente y los planos condicionan la construcción; "
                    "priorizar revisiones y levantamiento de observaciones.")
        else:
            nota = ("Actividades internas más controlables; mantener la "
                    "disciplina de aprobación de líneas base.")
        st.markdown(f"- **{fase}** (prob. media {r['Prob. retraso (%)']:.0f}%): {nota}")

    st.info(
        "El modelo es una herramienta de apoyo: prioriza y anticipa, pero la "
        "decisión sobre reprogramaciones y asignación de recursos corresponde "
        "al Jefe de Proyecto y su equipo."
    )

# ---------------------------------------------------------------------------
# TAB 3 · VISUALIZACIÓN
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Distribución de estados y riesgo")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicción enero 2026**")
        conteo_ene = df_res[df_res["Predicción enero 2026"] != "—"]["Predicción enero 2026"].value_counts()
        if len(conteo_ene):
            fig = px.pie(values=conteo_ene.values, names=conteo_ene.index,
                         color=conteo_ene.index, color_discrete_map=COLOR_CLASE, hole=0.45)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Predicción febrero 2026**")
        conteo_feb = df_res[df_res["Predicción febrero 2026"] != "—"]["Predicción febrero 2026"].value_counts()
        if len(conteo_feb):
            fig = px.pie(values=conteo_feb.values, names=conteo_feb.index,
                         color=conteo_feb.index, color_discrete_map=COLOR_CLASE, hole=0.45)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Probabilidad media de retraso por fase**")
    resumen = (df_res.groupby("Fase")["Prob. retraso (%)"].mean().reset_index()
               .sort_values("Prob. retraso (%)"))
    fig = px.bar(resumen, x="Prob. retraso (%)", y="Fase", orientation="h",
                 color="Prob. retraso (%)",
                 color_continuous_scale=["#2E9E5B", "#E8A200", "#C8102E"])
    fig.update_layout(coloraxis_showscale=False, height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 · METODOLOGÍA
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Cómo funciona el modelo")
    st.markdown(
        """
**Técnica:** clasificación supervisada multiclase con un **Random Forest**
(`scikit-learn`), que predice una de tres etiquetas por actividad y mes:
*A tiempo · Con riesgo de retraso · Retrasada*.

**Variables de entrada (features):** avance planificado, avance real, **SPI**
(Schedule Performance Index = avance real / avance planificado), rendimiento,
restricciones pendientes, disponibilidad de recursos, dependencias pendientes,
holgura, criticidad, retrasos previos y riesgo asociado a la fase.

**Predicción a dos meses:** para febrero se aplica un **arrastre de riesgo**:
si una actividad viene con riesgo o retraso en enero, sus condiciones de
febrero se degradan, reflejando la propagación de desviaciones a las
actividades sucesoras.

**Datos:** las **actividades son reales**, extraídas de la distribución mensual
del presupuesto del proyecto (partidas con avance programado en enero/febrero
2026). El avance y el desempeño se simulan de forma **reproducible** porque no
se dispone del registro de producción en vivo; para uso real, estas variables
se alimentan de Microsoft Project y los reportes de avance.
        """
    )
    m1, m2 = st.columns([1, 1])
    with m1:
        st.metric("Exactitud (test)", f"{ACC*100:.1f}%")
        st.metric("Registros de entrenamiento", len(df_train))
    with m2:
        st.markdown("**Desempeño por clase (F1-score):**")
        for clase in CLASES:
            if clase in REPORTE:
                st.markdown(f"- {clase}: {REPORTE[clase]['f1-score']:.2f}")

    st.markdown("**Importancia de las variables**")
    imp = pd.DataFrame({
        "Variable": FEATURES,
        "Importancia": MODELO.feature_importances_,
    }).sort_values("Importancia")
    fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                 color="Importancia", color_continuous_scale=["#9AB7D6", "#1A3E6E"])
    fig.update_layout(coloraxis_showscale=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.caption("Ejercicio · Predicción de cronograma con IA · Estación E4-8 Línea 4 Metro de Lima · v1.0")
