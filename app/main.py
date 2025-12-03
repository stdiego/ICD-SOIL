import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA

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
/* Fondo general suave */
.main > div {
    background: radial-gradient(circle at top left, #f0fdf4 0, #f9fafb 40%, #ffffff 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064e3b 0%, #065f46 40%, #022c22 100%);
    color: #ecfdf5 !important;
}

/* Títulos sidebar */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #ecfdf5 !important;
}

/* Botones */
button[kind="primary"] {
    border-radius: 999px !important;
    font-weight: 600 !important;
}

/* Tarjetas */
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
.ag-card-warn {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-color: #facc1533;
}
.ag-card-danger {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-color: #ef444433;
}

/* Etiquetas tipo pill */
.ag-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    border-radius: 999px;
    background: #dcfce7;
    color: #064e3b;
    margin-right: 0.4rem;
}

/* Footer */
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
    """
    Determina automáticamente cuál método usar (Olsen o Doble Ácido)
    según la categoría del cultivo.
    """

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

    # Cultivos donde ICA/AGROSAVIA prefieren Olsen
    cultivos_olsen = [
        "HORTALIZAS", "FRUTALES", "CAFÉ", "ARÁNDANO", "VID", "TOMATE",
        "MAÍZ", "PAPA", "FRESA", "CÍTRICOS", "ARROZ"
    ]

    if categoria.upper() in cultivos_olsen:
        return olsen[elemento]
    else:
        return doble_acido[elemento]

# ======================================================
# FUNCIÓN 2 — Alertas Agronómicas ICA / AGROSAVIA
# ======================================================
def detectar_alertas_avanzadas(df, categoria, valores):
    """
    Analiza relaciones catiónicas, acidez real, salinidad,
    deficiencias críticas y toxicidades basadas en ICA/AGROSAVIA.
    """

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

    # ---------------------------
    # Relación Ca/Mg
    # ---------------------------
    if Ca and Mg and Mg > 0:
        ratio = Ca / Mg
        if ratio < 2:
            alertas.append("⚠ Relación Ca/Mg baja (<2)")
            explicaciones.append("Puede causar compactación y baja estructura del suelo.")
        elif ratio > 8:
            alertas.append("⚠ Relación Ca/Mg alta (>8)")
            explicaciones.append("Puede causar bloqueo de Mg y desbalance catiónico.")

    # ---------------------------
    # Saturación de K: K/(Ca+Mg)
    # ---------------------------
    if Ca and Mg and K:
        base_sum = Ca + Mg
        if base_sum > 0:
            ratio_k = K / base_sum
            if ratio_k > 0.10:
                alertas.append("⚠ Saturación moderada de K (>10%)")
            if ratio_k > 0.15:
                alertas.append("🔥 Saturación severa de K (>15%)")

    # ---------------------------
    # Relación K/Mg
    # ---------------------------
    if K and Mg and Mg > 0:
        ratio_kmg = K / Mg
        if ratio_kmg > 0.30:
            alertas.append("⚠ Relación K/Mg desbalanceada (>0.30)")
        if ratio_kmg > 0.60:
            alertas.append("🔥 Relación K/Mg severamente alta (>0.60)")

    # ---------------------------
    # Saturación de acidez REAL: (H+Al + Al) / CIC
    # ---------------------------
    if H_Al and Al and CIC and CIC > 0:
        sat = (H_Al + Al) / CIC
        if sat > 0.60:
            alertas.append("🔥 Acidez severa — saturación > 60%")
        elif sat > 0.30:
            alertas.append("⚠ Acidez moderada — > 30%")
        elif sat > 0.15:
            alertas.append("⚠ Acidez ligera — > 15%")

    # ---------------------------
    # Toxicidad por Aluminio
    # ---------------------------
    if Al:
        if Al > 2:
            alertas.append("🔥 Toxicidad severa por Al (>2 cmol/kg)")
        elif Al > 1:
            alertas.append("⚠ Toxicidad ligera por Al (>1 cmol/kg)")

    # ---------------------------
    # Salinidad (CE)
    # ---------------------------
    if CE:
        if CE > 4:
            alertas.append("🔥 Salinidad severa (CE > 4 dS/m)")
        elif CE > 2:
            alertas.append("⚠ Salinidad moderada (CE > 2 dS/m)")

    # ---------------------------
    # Deficiencias críticas
    # ---------------------------
    if P and P < 10:
        alertas.append("⚠ Fósforo críticamente bajo (<10 mg/kg)")

    if B and B < 0.2:
        alertas.append("⚠ Boro críticamente bajo (<0.2 mg/kg)")

    return alertas, explicaciones

# ======================================================
# FUNCIÓN 3 — Recomendaciones Avanzadas
# ======================================================
def recomendaciones_avanzadas(alertas, categoria):
    recomendaciones = []

    for alerta in alertas:

        # Ca/Mg
        if "Ca/Mg baja" in alerta:
            recomendaciones.append("Aplicar cal dolomítica para mejorar Mg.")
        if "Ca/Mg alta" in alerta:
            recomendaciones.append("Aplicar yeso agrícola para equilibrar Ca sin subir pH.")

        # Saturación de K
        if "Saturación" in alerta:
            recomendaciones.append("Reducir aplicaciones de K y aumentar Ca/Mg si es necesario.")

        # Acidez
        if "Acidez" in alerta or "Aluminio" in alerta:
            recomendaciones.append("Aplicar encalado (CaCO3 o dolomita) fraccionado.")

        # Salinidad
        if "Salinidad" in alerta:
            recomendaciones.append("Mejorar drenaje y evitar fertilizantes salinos.")

        # Fósforo
        if "Fósforo" in alerta:
            recomendaciones.append("Aplicar MAP/DAP o fuentes fosfatadas según cultivo.")

        # Boro
        if "Boro" in alerta:
            recomendaciones.append("Aplicar B en dosis muy bajas (1–2 kg/ha).")

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

        st.markdown(
            """
<span class='ag-pill'>ICD automático</span>
<span class='ag-pill'>Detección de anomalías</span>
<span class='ag-pill'>Validación avanzada</span>
<span class='ag-pill'>Alertas ICA/AGROSAVIA</span>
<span class='ag-pill'>Forecast + PCA</span>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            """
Plataforma analítica que combina datos abiertos de suelos en Colombia
con algoritmos de IA y criterios agronómicos para:

- Evaluar la **calidad del dato (ICD)**
- Identificar **anomalías y registros sospechosos**
- Validar análisis recientes de laboratorio
- Explorar tendencias regionales y patrones multivariables
- Activar alertas agronómicas ICA/AGROSAVIA
            """
        )

        if st.button("🌍 Iniciar análisis", type="primary"):
            st.session_state["entered_app"] = True
            st.experimental_rerun()

    with col_w2:
        st.markdown(
            """
<div class="ag-card ag-card-success">
<b>🎯 Optimizado para el Concurso Datos al Ecosistema 2025</b><br><br>
<span style="font-size:0.9rem;">
Este panel integra ICD, anomalías, forecast, reglas ICA/AGROSAVIA y validación inteligente
para convertir análisis de suelo en recomendaciones confiables.
</span>
<br><br>
<ul style="font-size:0.85rem; color:#166534; padding-left:1.1rem;">
<li>ICD por variable, cultivo y ubicación</li>
<li>Alertas agronómicas automáticas</li>
<li>Validación avanzada por z-score y rangos ICA</li>
<li>Forecast + PCA multivariable</li>
</ul>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()

# ======================================================
# HEADER / HERO
# ======================================================
col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown("### 🌱 ICD Soil")
    st.markdown("## Calidad del dato y salud del suelo — Panel analítico completo")

    st.markdown(
        """
<span class='ag-pill'>ICD</span>
<span class='ag-pill'>Anomalías</span>
<span class='ag-pill'>Validación agronómica</span>
<span class='ag-pill'>Forecast</span>
<span class='ag-pill'>PCA</span>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Utiliza los filtros laterales para explorar la información por región, "
        "departamento, municipio y cultivo."
    )

with col_h2:
    st.markdown(
        """
<div class="ag-card ag-card-success">
<b>🧪 Dataset base</b><br>
<span style="font-size:0.9rem;">
Resultados de Análisis de Laboratorio Suelos en Colombia (datos abiertos).
</span>
<br><br>
<span style="font-size:0.85rem; color:#16a34a;">
El ICD combina completitud, anomalías y coherencia predictiva
para evaluar la confiabilidad del dato.
</span>
</div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.header("🎛 Filtros")

df_filtered = df.copy()

region = "Todas"
depto = "Todos"
muni = "Todos"
cultivo = "Todos"

# REGION
if "region" in df.columns:
    region = st.sidebar.selectbox(
        "🌍 Región:", ["Todas"] + sorted(df["region"].unique()),
        key="f_region"
    )
    if region != "Todas":
        df_filtered = df_filtered[df_filtered["region"] == region]

# DEPARTAMENTO
if group_col in df.columns:
    depto = st.sidebar.selectbox(
        "🏛 Departamento:", ["Todos"] + sorted(df[group_col].unique()),
        key="f_depto"
    )
    if depto != "Todos":
        df_filtered = df_filtered[df_filtered[group_col] == depto]

# MUNICIPIO

# CULTIVO
if "cultivo" in df_filtered.columns:
    cultivos_disp = ["Todos"] + sorted(df_filtered["cultivo"].unique())
    cultivo = st.sidebar.selectbox("🌾 Cultivo:", cultivos_disp, key="f_cultivo")

    if cultivo != "Todos":
        df_filtered = df_filtered[df_filtered["cultivo"] == cultivo]

# ======================================================
# VARIABLE & ICD
# ======================================================
variables = get_variables(df)
selected_var = st.selectbox("📌 Variable analizada:", variables, key="var_main")

var_icd_col = f"icd_total_{selected_var}"

st.subheader(f"📍 Resultado — {selected_var}")

if var_icd_col in df_filtered.columns:
    avg = df_filtered[var_icd_col].mean()
    st.metric("ICD promedio (filtros aplicados)", f"{avg:.3f}")
else:
    st.error(f"No existe `{var_icd_col}` en los datos.")

# KPI NACIONAL
if var_icd_col in df.columns:

    colA, colB, colC = st.columns(3)

    icd_nacional = df[var_icd_col].mean()
    colA.metric("ICD nacional", f"{icd_nacional:.3f}")

    # Mejor departamento
    if group_col in df.columns:
        icd_por_dep = df.groupby(group_col)[var_icd_col].mean().dropna()
        if not icd_por_dep.empty:
            mejor_dep = icd_por_dep.idxmax()
            mejor_val = icd_por_dep.max()
            colB.metric("Mejor departamento ICD", f"{mejor_dep}", f"{mejor_val:.3f}")

    # Porcentaje anomalías
    if "anom_score_global" in df_filtered.columns:
        porc_anom = (df_filtered["anom_score_global"] > 0.5).mean() * 100
        colC.metric("% registros sospechosos", f"{porc_anom:.1f}%")
    else:
        colC.metric("% registros sospechosos", "N/A")

# ======================================================
# FILTROS ACTIVOS
# ======================================================
active_filters = []

if region != "Todas":
    active_filters.append(f"Región: {region}")
if depto != "Todos":
    active_filters.append(f"Departamento: {depto}")
if muni != "Todos":
    active_filters.append(f"Municipio: {muni}")
if cultivo != "Todos":
    active_filters.append(f"Cultivo: {cultivo}")

if active_filters:
    st.info("🔎 Filtros aplicados → " + " | ".join(active_filters))

# ======================================================
# TABS PRINCIPALES
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🏅 Ranking",
        "🗺 Mapa",
        "📈 Forecast",
        "📊 Variabilidad",
        "⬇ Exportar",
        "🚨 Anomalías",
        "🧠 Perfil + Guía ICD",
        "🧪 Validar muestra",
    #    "📉 PCA multivariable",
    ]
)

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
            title=f"Ranking ICD — {selected_var}",
        )

        st.plotly_chart(fig_rank, use_container_width=True)
        st.dataframe(ranking, use_container_width=True)

    else:
        st.warning("No hay datos disponibles para construir el ranking.")

# ======================================================
# TAB 2 — MAPA
# ======================================================
with tab2:
    st.subheader(f"🗺 Mapa ICD — {selected_var}")

    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        geo_match = None

        # Buscar columna coincidente (DEPTO)
        for col in gdf.columns:
            if col.lower() != "geometry":
                if set(df[group_col].unique()).intersection(
                        set(gdf[col].astype(str).str.upper())
                ):
                    geo_match = col
                    gdf[col] = gdf[col].astype(str).str.upper()
                    break

        if geo_match is None:
            st.error(
                "No hay coincidencia entre GeoJSON y nombres de departamento."
            )
        else:
            ranking_geo = (
                df_filtered.groupby(group_col)[var_icd_col]
                .mean()
                .reset_index()
            )

            merged = gdf.merge(
                ranking_geo,
                left_on=geo_match,
                right_on=group_col,
                how="left",
            )

            fig_map = px.choropleth_mapbox(
                merged,
                geojson=merged.geometry,
                locations=merged.index,
                color=var_icd_col,
                hover_name=geo_match,
                mapbox_style="carto-positron",
                color_continuous_scale="Viridis",
                zoom=4.5,
                center={"lat": 4.5, "lon": -74},
                opacity=0.85,
                height=600,
            )
            st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"Error al cargar el mapa → {e}")

# ======================================================
# TAB 3 — FORECAST
# ======================================================
with tab3:
    st.subheader("📈 Forecast (histórico + pronóstico + carta de control)")

    if df_forecast is not None:
        df_fx = df_forecast[df_forecast["variable"] == selected_var].copy()

        if df_fx.empty:
            st.warning("No hay forecast disponible para esta variable.")
        else:
            df_fx["fecha"] = pd.to_datetime(df_fx["fecha"], errors="coerce")
            df_fx = df_fx.dropna(subset=["fecha"]).sort_values("fecha")

            # Tipos de serie
            if "tipo" in df_fx.columns:
                df_hist = df_fx[df_fx["tipo"].str.contains("hist", case=False)]
                df_pred = df_fx[df_fx["tipo"].str.contains("fore", case=False)]
            else:
                df_hist = df_fx.copy()
                df_pred = pd.DataFrame()

            fig_fx = px.line(
                df_fx,
                x="fecha",
                y="valor",
                color="tipo" if "tipo" in df_fx.columns else None,
                markers=True,
                title=f"Serie histórica y pronóstico — {selected_var}",
            )

            # Carta de control
            if not df_hist.empty:
                mu = df_hist["valor"].mean()
                sigma = df_hist["valor"].std(ddof=1)

                ucl = mu + 3 * sigma
                lcl = mu - 3 * sigma

                fig_fx.add_hline(y=mu, line_dash="dot", annotation_text="Media")
                fig_fx.add_hline(y=ucl, line_dash="dash", annotation_text="UCL")
                fig_fx.add_hline(y=lcl, line_dash="dash", annotation_text="LCL")

            st.plotly_chart(fig_fx, use_container_width=True)

            # Interpretación automática
            if not df_hist.empty:
                st.write("---")
                st.markdown("### 🔎 Interpretación automática del forecast")

                valores_hist = df_hist["valor"].values
                delta = valores_hist[-1] - valores_hist[0]

                # Tendencia
                if sigma == 0:
                    tendencia = "Serie estable sin variación."
                elif abs(delta) <= 0.1 * sigma:
                    tendencia = "Serie estable sin tendencia clara."
                elif delta > 0:
                    tendencia = "Tendencia levemente ascendente."
                else:
                    tendencia = "Tendencia levemente descendente."

                # Variabilidad
                mu_safe = mu if mu != 0 else 1e-9
                cv = abs(sigma / mu_safe)
                if cv < 0.10:
                    var_txt = "Baja variabilidad."
                elif cv < 0.25:
                    var_txt = "Variabilidad media."
                else:
                    var_txt = "Alta variabilidad."

                # Riesgo
                fuera_control = (
                        (df_hist["valor"] > ucl) | (df_hist["valor"] < lcl)
                ).mean()

                if fuera_control == 0:
                    riesgo_hist = "Sin valores fuera de control."
                elif fuera_control < 0.05:
                    riesgo_hist = "Bajo riesgo, pocos valores fuera de control."
                else:
                    riesgo_hist = "Riesgo alto: múltiples valores fuera de los límites."

                st.markdown(
                    f"""
- **Tendencia:** {tendencia}  
- **Variabilidad:** {var_txt}  
- **Riesgo histórico:** {riesgo_hist}  
                    """
                )
    else:
        st.info("No existe archivo de forecast.")

# ======================================================
# TAB 4 — VARIABILIDAD (BOXPLOT)
# ======================================================
with tab4:
    st.subheader("📊 Variabilidad por Departamento")

    if var_icd_col in df_filtered.columns:
        fig_box = px.box(
            df_filtered,
            x=group_col,
            y=var_icd_col,
            title=f"Variabilidad del ICD — {selected_var}",
        )
        fig_box.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No hay suficiente información para evaluar la variabilidad.")

# ======================================================
# TAB 5 — EXPORTACIÓN
# ======================================================
with tab5:
    st.subheader("⬇ Exportar datos filtrados")

    if var_icd_col in df_filtered.columns:
        cols_export = [
            c
            for c in [group_col, "region", "municipio", "cultivo", var_icd_col]
            if c in df_filtered.columns
        ]

        csv_bytes = (
            df_filtered[cols_export]
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "📥 Descargar CSV filtrado",
            csv_bytes,
            file_name=f"ICD_{selected_var}_filtrado.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay datos disponibles para exportar con los filtros actuales.")

# ======================================================
# TAB 6 — ANOMALÍAS
# ======================================================
with tab6:
    st.subheader("🚨 Anomalías detectadas")

    if "anom_score_global" in df_filtered.columns:
        df_anom = df_filtered.sort_values(
            "anom_score_global", ascending=False
        ).head(50)

        # Columna para el eje x
        if selected_var in df_filtered.columns:
            x_col = selected_var
        else:
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            x_col = numeric_cols[0] if len(numeric_cols) else None

        if x_col is not None:
            fig_anom = px.scatter(
                df_filtered,
                x=x_col,
                y="anom_score_global",
                color="anom_score_global",
                color_continuous_scale="Reds",
                title=f"Distribución de anomalías — {x_col}",
                hover_data=["cultivo", group_col, "municipio"],
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        st.write("🔎 **Top 50 registros más sospechosos:**")
        st.dataframe(df_anom.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No existe columna 'anom_score_global' en los datos.")

# ======================================================
# TAB 7 — PERFIL AGRONÓMICO
# ======================================================
with tab7:
    st.subheader("🧠 Perfil agronómico del suelo (según filtros aplicados)")

    if df_filtered.empty:
        st.info("No hay datos para los filtros seleccionados.")
    else:
        # Contexto
        ctx = []
        if region != "Todas": ctx.append(f"Región: `{region}`")
        if depto != "Todos": ctx.append(f"Departamento: `{depto}`")
        if muni != "Todos": ctx.append(f"Municipio: `{muni}`")
        if cultivo != "Todos": ctx.append(f"Cultivo: `{cultivo}`")
        if not ctx: ctx.append("Perfil nacional sin filtros.")
        st.markdown(" · ".join(ctx))

        st.write("---")
        st.markdown("### 📊 Variables clave del suelo (promedio filtrado)")

        vars_clave = [
            "ph_agua_suelo",
            "materia_organica",
            "fosforo_bray_ii",
            "conductividad_electrica",
            "cic",
        ]

        presentes = [v for v in vars_clave if v in df_filtered.columns]

        resumen = []

        for v in presentes:
            serie = df_filtered[v].dropna()
            if serie.empty: continue

            valor = float(serie.mean())

            # Reglas visuales
            if v == "ph_agua_suelo":
                if valor < 5.5:
                    estado = "Ácido — riesgo de baja disponibilidad."
                elif valor > 7.5:
                    estado = "Alcalino — posible bloqueo de micronutrientes."
                else:
                    estado = "Óptimo."

            elif v == "materia_organica":
                if valor < 2:
                    estado = "Baja — mejorar con enmiendas orgánicas."
                elif valor <= 4:
                    estado = "Media — aceptable."
                else:
                    estado = "Alta — buen nivel."

            elif v == "fosforo_bray_ii":
                if valor < 15:
                    estado = "Bajo — limita desarrollo radicular."
                elif valor <= 30:
                    estado = "Adecuado."
                else:
                    estado = "Alto — revisar dosis fosfatada."

            elif v == "conductividad_electrica":
                if valor < 2:
                    estado = "Sin problemas."
                elif valor <= 4:
                    estado = "Salinidad moderada."
                else:
                    estado = "Salinidad severa."

            elif v == "cic":
                if valor < 10:
                    estado = "Baja — suelo ligero."
                elif valor <= 25:
                    estado = "Media — buena capacidad."
                else:
                    estado = "Alta — excelente retención."

            resumen.append(
                {
                    "Variable": v,
                    "Promedio": round(valor, 2),
                    "Interpretación": estado,
                }
            )

        if resumen:
            st.dataframe(pd.DataFrame(resumen), use_container_width=True)


# ======================================================
# TAB 8 — VALIDACIÓN AGRONÓMICA INTELIGENTE
# ======================================================
with tab8:
    st.subheader("🧪 Validación agronómica inteligente")
    st.write("Ingresa valores medidos. El sistema compara con cultivo, ubicación y rangos ICA/AGROSAVIA.")

    variables_ingreso = st.multiselect("Variables a validar:", variables, key="v_ingreso")

    if variables_ingreso:

        # === 1) Reglas agronómicas ICA/AGROSAVIA ===
        reglas = {
            "calcio_intercambiable":     {"crit": 2,    "bajo": 4,   "alto": 10},
            "magnesio_intercambiable":   {"crit": 0.5,  "bajo": 1,   "alto": 3},
            "potasio_intercambiable":    {"crit": 0.1,  "bajo": 0.3, "alto": 1},
            "sodio_intercambiable":      {"crit": 0.1,  "bajo": 0.3, "alto": 1},
            "fosforo_bray_ii":           {"crit": 10,   "bajo": 20,  "alto": 40},
            "azufre_fosfato_monocalcico":{"crit": 5,    "bajo": 10,  "alto": 20},
            "boro_disponible":           {"crit": 0.1,  "bajo": 0.3, "alto": 1},
            "zinc_disponible_olsen":     {"crit": 0.5,  "bajo": 1,   "alto": 5},
            "cobre_disponible":          {"crit": 0.2,  "bajo": 0.5, "alto": 5},
            "manganeso_disponible_olsen":{"crit": 2,    "bajo": 5,   "alto": 50},
            "hierro_disponible_olsen":   {"crit": 2,    "bajo": 5,   "alto": 20},
            "acidez_intercambiable":     {"crit": 3,    "bajo": 2,   "alto": 1e9},
            "aluminio_intercambiable":   {"crit": 2,    "bajo": 1,   "alto": 1e9},
        }

        valores = {}
        for v in variables_ingreso:
            valores[v] = st.number_input(f"{v}:", value=0.0, key=f"in_{v}")

        if st.button("🔍 Evaluar muestra", key="eval_button_new"):
            resultados = []
            alertas = []
            icd_scores = []

            for v, x in valores.items():

                # Default
                nivel = "Excelente"
                icd = 0.95
                rec = "✔ Valores dentro del rango esperado."

                if v in reglas:
                    r = reglas[v]

                    if x < r["crit"]:
                        nivel = "Crítico"
                        icd = 0.20
                        rec = "🛑 Valor crítico. Afectará gravemente el cultivo."
                        alertas.append(f"❌ {v}: nivel crítico.")
                    elif x < r["bajo"]:
                        nivel = "Bajo"
                        icd = 0.45
                        rec = "⚠ Nivel bajo, requiere corrección."
                        alertas.append(f"⚠ {v}: nivel bajo.")
                    elif x > r["alto"]:
                        nivel = "Alto / Riesgo"
                        icd = 0.50
                        rec = "⚠ Nivel alto, posible toxicidad."
                        alertas.append(f"⚠ {v}: nivel alto.")
                    else:
                        nivel = "Adecuado"
                        icd = 0.90
                        rec = "✔ Nivel adecuado para la mayoría de cultivos."

                resultados.append((v, x, nivel, icd, rec))
                icd_scores.append(icd)

            # === Mostrar resultados ===
            for v, x, nivel, icd, rec in resultados:
                st.markdown(f"""
                <div class="ag-card">
                <h4>{v} — {nivel}</h4>
                Valor ingresado: <b>{x}</b><br>
                ICD ajustado: <b>{icd:.2f}</b><br>
                <b>Recomendación:</b> {rec}
                </div>
                """, unsafe_allow_html=True)

            # === Alertas ===
            st.write("---")
            st.subheader("🚨 Alertas agronómicas (ICA/AGROSAVIA)")

            if alertas:
                for a in alertas:
                    st.error(a)
            else:
                st.success("🟢 No se detectaron alertas agronómicas relevantes.")

            # === ICD global ===
            icd_global = sum(icd_scores) / len(icd_scores)
            st.write("---")
            st.subheader("📈 Resultado global del análisis")

            if icd_global >= 0.90:
                st.success(f"🟢 Excelente — ICD global: {icd_global:.2f}")
            elif icd_global >= 0.75:
                st.info(f"🔵 Bueno — ICD global: {icd_global:.2f}")
            elif icd_global >= 0.60:
                st.warning(f"🟡 Moderado — ICD global: {icd_global:.2f}")
            elif icd_global >= 0.40:
                st.error(f"🟠 Alto riesgo — ICD global: {icd_global:.2f}")
            else:
                st.error(f"🔴 Crítico — ICD global: {icd_global:.2f}")

# ======================================================
# FOOTER
# ======================================================
st.markdown(
    "<div class='ag-footer'>✔ ICD Soil — Plataforma avanzada de calidad del dato, alertas y forecast.</div>",
    unsafe_allow_html=True,
)

