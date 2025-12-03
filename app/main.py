import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json

from utils import load_data, get_variables, get_group_col, GEOJSON_PATH

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
st.set_page_config(page_title="ICD Soil", layout="wide")

# ======================================================
# CSS GLOBAL — Tema Moderno
# ======================================================
APP_CSS = """
<style>
.main > div {
    background: radial-gradient(circle at top left, #f0fdf4 0, #f9fafb 40%, #ffffff 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064e3b 0%, #065f46 40%, #022c22 100%);
    color: #ecfdf5 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #ecfdf5 !important;
}
button[kind="primary"] {
    border-radius: 999px !important;
    font-weight: 600 !important;
}
.ag-card {
    padding: 1rem 1.2rem;
    border-radius: 1rem;
    background: #ffffff;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.ag-card-success {
    background: linear-gradient(135deg, #ecfdf5, #dcfce7);
    border-color: #22c55e33;
}
.ag-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    border-radius: 999px;
    background: #dcfce7;
    color: #064e3b;
    margin-right: 0.4rem;
}
.ag-footer {
    font-size: 0.8rem;
    color: #6b7280;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ======================================================
# CARGA DE DATOS
# ======================================================
df, df_depto, df_forecast = load_data()

if df is None:
    st.stop()

group_col = get_group_col(df) or "DEPARTAMENTO"
variables = get_variables(df)

# Normalizar textos clave
for col in ["departamento", "dep_norm", "region", "municipio", group_col, "cultivo"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()

# ======================================================
# FUNCIÓN 1 — Selección de método (Olsen vs Doble Ácido)
# ======================================================
def seleccionar_micronutriente(categoria, elemento):
    olsen = {
        "Fe": "hierro_disponible_olsen",
        "Mn": "manganeso_disponible_olsen",
        "Zn": "zinc_disponible_olsen",
        "Cu": "cobre_disponible",
    }
    doble_acido = {
        "Fe": "hierro_disponible_doble_acido",
        "Mn": "manganeso_disponible_doble_acido",
        "Zn": "zinc_disponible_doble_acido",
        "Cu": "cobre_disponible_doble_acido",
    }
    cultivos_olsen = [
        "HORTALIZAS", "FRUTALES", "CAFÉ", "ARÁNDANO", "VID", "TOMATE",
        "MAÍZ", "PAPA", "FRESA", "CÍTRICOS", "ARROZ"
    ]
    return olsen[elemento] if categoria.upper() in cultivos_olsen else doble_acido[elemento]

# ======================================================
# FUNCIÓN 2 — Alertas ICA / AGROSAVIA
# ======================================================
def detectar_alertas_avanzadas(df, categoria, valores):
    alertas = []
    explicaciones = []

    Ca = valores.get("calcio_intercambiable")
    Mg = valores.get("magnesio_intercambiable")
    K  = valores.get("potasio_intercambiable")
    H_Al = valores.get("acidez_intercambiable")
    Al = valores.get("aluminio_intercambiable")
    CIC = valores.get("capacidad_de_intercambio_cationico")
    CE = valores.get("conductividad_electrica")
    P = valores.get("fosforo_bray_ii")
    B = valores.get("boro_disponible")

    if Ca and Mg and Mg > 0:
        ratio = Ca / Mg
        if ratio < 2:
            alertas.append("⚠ Relación Ca/Mg baja (<2)")
        elif ratio > 8:
            alertas.append("⚠ Relación Ca/Mg alta (>8)")

    if Ca and Mg and K:
        base_sum = Ca + Mg
        if base_sum > 0:
            ratio_k = K / base_sum
            if ratio_k > 0.10:
                alertas.append("⚠ Saturación moderada de K (>10%)")
            if ratio_k > 0.15:
                alertas.append("🔥 Saturación severa de K (>15%)")

    if K and Mg and Mg > 0:
        ratio_kmg = K / Mg
        if ratio_kmg > 0.30:
            alertas.append("⚠ Relación K/Mg desbalanceada (>0.30)")
        if ratio_kmg > 0.60:
            alertas.append("🔥 Relación K/Mg severamente alta (>0.60)")

    if H_Al and Al and CIC and CIC > 0:
        sat = (H_Al + Al) / CIC
        if sat > 0.60:
            alertas.append("🔥 Acidez severa — saturación > 60%")
        elif sat > 0.30:
            alertas.append("⚠ Acidez moderada — > 30%")

    if Al:
        if Al > 2:
            alertas.append("🔥 Toxicidad severa por Al (>2 cmol/kg)")
        elif Al > 1:
            alertas.append("⚠ Toxicidad ligera por Al (>1 cmol/kg)")

    if CE:
        if CE > 4:
            alertas.append("🔥 Salinidad severa (CE > 4 dS/m)")
        elif CE > 2:
            alertas.append("⚠ Salinidad moderada (CE > 2 dS/m)")

    if P and P < 10:
        alertas.append("⚠ Fósforo críticamente bajo (<10 mg/kg)")

    if B and B < 0.2:
        alertas.append("⚠ Boro críticamente bajo (<0.2 mg/kg)")

    return alertas, explicaciones

# ======================================================
# FUNCIÓN 3 — Recomendaciones
# ======================================================
def recomendaciones_avanzadas(alertas, categoria):
    recomendaciones = []
    for alerta in alertas:
        if "Ca/Mg baja" in alerta:
            recomendaciones.append("Aplicar cal dolomítica.")
        if "Ca/Mg alta" in alerta:
            recomendaciones.append("Aplicar yeso agrícola.")
        if "Saturación" in alerta:
            recomendaciones.append("Reducir aplicaciones de K.")
        if "Acidez" in alerta or "Aluminio" in alerta:
            recomendaciones.append("Aplicar encalado gradual.")
        if "Salinidad" in alerta:
            recomendaciones.append("Mejorar drenaje.")
        if "Fósforo" in alerta:
            recomendaciones.append("Aplicar fosfatos.")
        if "Boro" in alerta:
            recomendaciones.append("Aplicar B en dosis bajas.")
    return recomendaciones

# ======================================================
# WELCOME SCREEN
# ======================================================
if "entered_app" not in st.session_state:
    st.session_state["entered_app"] = False

if not st.session_state["entered_app"]:
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        st.markdown("## 🌱 ICD Soil")
        st.markdown("### Inteligencia para la salud del suelo")
        st.markdown("""
<span class='ag-pill'>ICD automático</span>
<span class='ag-pill'>Detección de anomalías</span>
<span class='ag-pill'>Validación avanzada</span>
<span class='ag-pill'>Alertas ICA/AGROSAVIA</span>
<span class='ag-pill'>Forecast + PCA</span>
        """, unsafe_allow_html=True)

        st.write("""
Plataforma analítica que combina datos abiertos con IA para:
- Evaluar **calidad del dato (ICD)**
- Detectar **anomalías**
- Validar muestras de laboratorio
- Activar alertas ICA/AGROSAVIA
- Explorar patrones multivariables
        """)

        if st.button("🌍 Iniciar análisis", type="primary"):
            st.session_state["entered_app"] = True
            st.experimental_rerun()

    with col_w2:
        st.markdown("""
<div class="ag-card ag-card-success">
<b>🎯 Optimizado para Datos al Ecosistema 2025</b><br><br>
Incluye ICD, anomalías, forecast y validación ICA/AGROSAVIA.
</div>
        """, unsafe_allow_html=True)

    st.stop()

# ======================================================
# HEADER
# ======================================================
st.markdown("### 🌱 ICD Soil — Panel Analítico")
st.markdown("#### Calidad del dato, anomalías, forecast y validación agronómica")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("🎛 Filtros")

df_filtered = df.copy()
region = depto = muni = cultivo = "Todos"

if "region" in df.columns:
    region = st.sidebar.selectbox("🌍 Región:", ["Todos"] + sorted(df["region"].unique()))
    if region != "Todos":
        df_filtered = df_filtered[df_filtered["region"] == region]

if group_col in df.columns:
    depto = st.sidebar.selectbox("🏛 Departamento:", ["Todos"] + sorted(df[group_col].unique()))
    if depto != "Todos":
        df_filtered = df_filtered[df_filtered[group_col] == depto]

if "cultivo" in df_filtered.columns:
    cultivo = st.sidebar.selectbox("🌾 Cultivo:", ["Todos"] + sorted(df_filtered["cultivo"].unique()))
    if cultivo != "Todos":
        df_filtered = df_filtered[df_filtered["cultivo"] == cultivo]

# ======================================================
# VARIABLE & ICD
# ======================================================
variables = get_variables(df)
selected_var = st.selectbox("📌 Variable analizada:", variables)

var_icd_col = f"icd_total_{selected_var}"

st.subheader(f"📍 Resultado — {selected_var}")

if var_icd_col in df_filtered.columns:
    st.metric("ICD promedio", f"{df_filtered[var_icd_col].mean():.3f}")
else:
    st.error(f"No existe `{var_icd_col}` en los datos.")

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏅 Ranking",
    "🗺 Mapa",
    "📈 Forecast",
    "📊 Variabilidad",
    "⬇ Exportar",
    "🚨 Anomalías",
    "🧠 Perfil ICD",
    "🧪 Validación",
])

# ======================================================
# TAB 1 — RANKING
# ======================================================
with tab1:
    st.subheader(f"🏅 Ranking ICD — {selected_var}")
    if var_icd_col in df_filtered.columns:
        ranking = (
            df_filtered.groupby(group_col)[var_icd_col]
            .mean()
            .reset_index()
            .sort_values(var_icd_col, ascending=False)
        )
        fig_rank = px.bar(
            ranking,
            x=var_icd_col,
            y=group_col,
            orientation="h",
            color=var_icd_col,
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        st.dataframe(ranking)

# ======================================================
# TAB 2 — MAPA SIN GEOPANDAS (COMPATIBLE CLOUD)
# ======================================================
with tab2:
    st.subheader(f"🗺 Mapa ICD — {selected_var}")

    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        sample_props = geojson["features"][0]["properties"]
        posibles_keys = list(sample_props.keys())

        candidatos = [
            "NOMBRE_DPT", "DPTO_CNMBR", "NOMBRE_DEP",
            "DEPARTAMENTO", "departamento", "name", "NAME"
        ]

        match_key = next((c for c in candidatos if c in posibles_keys), None)

        if not match_key:
            st.error("No se encontró propiedad de departamento en GeoJSON.")
            st.stop()

        ranking_geo = (
            df_filtered.groupby(group_col)[var_icd_col]
            .mean()
            .reset_index()
        )
        ranking_geo[group_col] = ranking_geo[group_col].astype(str).str.upper()

        fig_map = px.choropleth(
            ranking_geo,
            geojson=geojson,
            locations=group_col,
            featureidkey=f"properties.{match_key}",
            color=var_icd_col,
            color_continuous_scale="Viridis",
            projection="mercator",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Error generando mapa: {e}")

# ======================================================
# (Las demás pestañas siguen igual que tu versión final)
# ======================================================

st.markdown("<div class='ag-footer'>✔ ICD Soil — Plataforma avanzada de calidad del dato, alertas y forecast.</div>", unsafe_allow_html=True)
