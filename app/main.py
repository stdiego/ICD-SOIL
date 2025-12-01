import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA

from utils import load_data, get_variables, get_group_col, GEOJSON_PATH

# ====================== CONFIG ======================
st.set_page_config(page_title="ICD Soil", layout="wide")

# ====================== THEME / CSS GLOBAL ======================
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

/* Tarjetas personalizadas */
.ag-card {
    padding: 1rem 1.2rem;
    border-radius: 1rem;
    background: #ffffff;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

/* Variantes de tarjetas */
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

/* Pequeño footer suave */
.ag-footer {
    font-size: 0.8rem;
    color: #6b7280;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# ====================== LOAD DATA ======================
df, df_depto, df_forecast = load_data()

group_col = get_group_col(df) or "DEPARTAMENTO"
variables = get_variables(df)

# Normalizar nombres
for col in ["departamento", "dep_norm", "region", "municipio", group_col, "cultivo"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()

# ====================== WELCOME SCREEN (LANDING) ======================
if "entered_app" not in st.session_state:
    st.session_state["entered_app"] = False

if not st.session_state["entered_app"]:
    col_w1, col_w2 = st.columns([2, 1])

    with col_w1:
        st.markdown("## 🌱 ICD Soil")
        st.markdown("### Inteligencia para la salud del suelo")
        st.markdown(
            "<span class='ag-pill'>ICD automático</span>"
            "<span class='ag-pill'>Detección de anomalías</span>"
            "<span class='ag-pill'>Validación de análisis</span>"
            "<span class='ag-pill'>PCA + Forecast</span>",
            unsafe_allow_html=True,
        )
        st.write(
            "Plataforma analítica que combina datos abiertos de suelos en Colombia, "
            "modelos de IA y criterios agronómicos para evaluar la calidad del dato, "
            "detectar anomalías y apoyar decisiones de fertilización y manejo."
        )
        st.write(
            "- 📊 Índice de Calidad del Dato (ICD)
"
            "- 🚨 Detección de valores atípicos
"
            "- 🧪 Validación inteligente de nuevos análisis
"
            "- 🤖 Forecast y análisis multivariable (PCA)"
        )

        if st.button("🌍 Explorar datos", type="primary"):
            st.session_state["entered_app"] = True
            st.experimental_rerun()

    with col_w2:
        st.markdown(
            """
            <div class="ag-card ag-card-success">
                <b>🎯 Listo para el reto</b><br>
                <span style="font-size:0.9rem;">
                Proyecto desarrollado para el Concurso Datos al Ecosistema 2025.
                Integra calidad de dato, anomalías y analítica avanzada en un solo panel.
                </span>
                <br><br>
                <ul style="font-size:0.85rem; color:#166534; padding-left:1.1rem;">
                    <li>ICD por variable, cultivo y territorio</li>
                    <li>Alertas tempranas de registros sospechosos</li>
                    <li>Validación estadística de resultados de laboratorio</li>
                    <li>Perfil agronómico según filtros aplicados</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()

# ====================== HEADER / HERO ======================
col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown("### 🌱 ICD Soil")
    st.markdown("## Calidad del dato y salud del suelo en un solo vistazo")
    st.markdown(
        "<span class='ag-pill'>ICD por variable</span>"
        "<span class='ag-pill'>Anomalías</span>"
        "<span class='ag-pill'>Forecast</span>"
        "<span class='ag-pill'>PCA multivariable</span>",
        unsafe_allow_html=True,
    )
    st.write(
        "Filtra por región, departamento, municipio y cultivo para explorar cómo se "
        "comporta la calidad del dato en diferentes contextos productivos."
    )

with col_h2:
    st.markdown(
        """
        <div class="ag-card ag-card-success">
            <b>🧪 Dataset</b><br>
            <span style="font-size:0.9rem;">
            Resultados de Análisis de Laboratorio Suelos en Colombia (datos abiertos).
            </span><br><br>
            <span style="font-size:0.85rem; color:#16a34a;">
            ICD Soil combina completitud, anomalías y coherencia predictiva
            para generar un índice único de calidad del dato.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====================== SIDEBAR FILTERS ======================
st.sidebar.header("🎛 Filtros")

df_filtered = df.copy()

region = "Todas"
depto = "Todos"
muni = "Todos"
cultivo = "Todos"

if "region" in df.columns:
    region = st.sidebar.selectbox("🌍 Región:", ["Todas"] + sorted(df["region"].unique()), key="f_region")
    if region != "Todas":
        df_filtered = df_filtered[df_filtered["region"] == region]

if group_col in df.columns:
    depto = st.sidebar.selectbox("🏛 Departamento:", ["Todos"] + sorted(df[group_col].unique()), key="f_depto")
    if depto != "Todos":
        df_filtered = df_filtered[df_filtered[group_col] == depto]

if "municipio" in df.columns:
    muni = st.sidebar.selectbox("📍 Municipio:", ["Todos"] + sorted(df["municipio"].unique()), key="f_muni")
    if muni != "Todos":
        df_filtered = df_filtered[df_filtered["municipio"] == muni]

# Cultivo filtrado por filtros previos
if "cultivo" in df_filtered.columns:
    cultivos_disp = ["Todos"] + sorted(df_filtered["cultivo"].unique())
    cultivo = st.sidebar.selectbox("🌾 Cultivo:", cultivos_disp, key="f_cultivo")
    if cultivo != "Todos":
        df_filtered = df_filtered[df_filtered["cultivo"] == cultivo]

# ====================== VARIABLE SELECCIONADA ======================
selected_var = st.selectbox("📌 Variable analizada:", variables, key="var_main")
var_icd_col = f"icd_total_{selected_var}"

# ====================== METRICAS PRINCIPALES ======================
st.subheader(f"📍 Resultado — {selected_var}")

if var_icd_col in df_filtered.columns:
    avg = df_filtered[var_icd_col].mean()
    st.metric("ICD promedio (filtros aplicados)", f"{avg:.3f}")
else:
    st.error(f"No existe `{var_icd_col}` en los datos.")

# KPIs adicionales a nivel general
if var_icd_col in df.columns:
    colA, colB, colC = st.columns(3)

    icd_nacional = df[var_icd_col].mean()
    colA.metric("ICD nacional", f"{icd_nacional:.3f}")

    if group_col in df.columns:
        icd_por_dep = df.groupby(group_col)[var_icd_col].mean().dropna()
        if not icd_por_dep.empty:
            mejor_dep = icd_por_dep.idxmax()
            mejor_val = icd_por_dep.max()
            colB.metric("Mejor departamento ICD", f"{mejor_dep}", f"{mejor_val:.3f}")

    if "anom_score_global" in df.columns:
        porc_anom = (df["anom_score_global"] > 0.5).mean() * 100
        colC.metric("% registros sospechosos", f"{porc_anom:.1f}%")

# ====================== FILTROS ACTIVOS ======================
active_filters = []
if "region" in df.columns and region != "Todas":
    active_filters.append(f"Región: {region}")
if depto != "Todos":
    active_filters.append(f"Departamento: {depto}")
if muni != "Todos":
    active_filters.append(f"Municipio: {muni}")
if cultivo != "Todos":
    active_filters.append(f"Cultivo: {cultivo}")

if active_filters:
    st.info("🔎 Filtros aplicados → " + " | ".join(active_filters))

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "🏅 Ranking",
        "🗺 Mapa",
        "📈 Forecast",
        "📊 Variabilidad",
        "⬇ Exportar",
        "🚨 Anomalías",
        "🧠 Perfil + Guía ICD",
        "🧪 Validar muestra",
        "📉 PCA multivariable",
    ]
)

# ====================== TAB 1 — RANKING ======================
with tab1:
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
        st.warning("No hay datos suficientes para construir el ranking.")

# ====================== TAB 2 — MAPA ======================
with tab2:
    st.subheader(f"🗺 Mapa ICD — {selected_var}")
    try:
        gdf = gpd.read_file(GEOJSON_PATH)
        geo_match = None

        for col in gdf.columns:
            if col.lower() != "geometry":
                if set(df[group_col].unique()).intersection(
                    set(gdf[col].astype(str).str.upper())
                ):
                    geo_match = col
                    gdf[col] = gdf[col].astype(str).str.upper()
                    break

        if geo_match is None:
            st.error("No hay coincidencia entre el GeoJSON y los nombres de departamento.")
        else:
            ranking_geo = df_filtered.groupby(group_col)[var_icd_col].mean().reset_index()
            merged = gdf.merge(
                ranking_geo, left_on=geo_match, right_on=group_col, how="left"
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
        st.error(f"Error cargando mapa → {e}")

# ====================== TAB 3 — FORECAST ======================
with tab3:
    st.subheader("📈 Forecast (serie + carta de control simple)")

    if df_forecast is not None:
        df_fx = df_forecast[df_forecast["variable"] == selected_var].copy()

        if df_fx.empty:
            st.warning("No hay forecast disponible para esta variable.")
        else:
            df_fx["fecha"] = pd.to_datetime(df_fx["fecha"], errors="coerce")
            df_fx = df_fx.dropna(subset=["fecha"]).sort_values("fecha")

            # Histórico vs pronóstico si 'tipo' está disponible
            if "tipo" in df_fx.columns:
                df_hist = df_fx[df_fx["tipo"].str.contains("hist", case=False, na=False)].copy()
                df_pred = df_fx[df_fx["tipo"].str.contains("fore", case=False, na=False)].copy()
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

            # Carta de control simple sobre el histórico
            if not df_hist.empty:
                mu = df_hist["valor"].mean()
                sigma = df_hist["valor"].std(ddof=1)
                ucl = mu + 3 * sigma
                lcl = mu - 3 * sigma

                fig_fx.add_hline(
                    y=mu,
                    line_dash="dot",
                    annotation_text="CL (media)",
                    annotation_position="top left",
                )
                fig_fx.add_hline(
                    y=ucl,
                    line_dash="dash",
                    annotation_text="UCL (μ + 3σ)",
                    annotation_position="top left",
                )
                fig_fx.add_hline(
                    y=lcl,
                    line_dash="dash",
                    annotation_text="LCL (μ - 3σ)",
                    annotation_position="bottom left",
                )

            fig_fx.update_layout(xaxis_title="Fecha", yaxis_title=selected_var)
            st.plotly_chart(fig_fx, use_container_width=True)

            # Interpretación automática
            if not df_hist.empty:
                st.write("---")
                st.markdown("### 🔎 Interpretación automática del forecast")

                valores_hist = df_hist["valor"].values
                mu = float(mu)
                sigma = float(sigma) if not np.isnan(sigma) else 0.0
                delta = float(valores_hist[-1] - valores_hist[0])

                # Tendencia
                if sigma == 0:
                    tendencia = "Estable (sin variación apreciable en la serie histórica)."
                else:
                    if abs(delta) <= 0.1 * sigma:
                        tendencia = "Estable: no se observa tendencia clara creciente o decreciente."
                    elif delta > 0:
                        tendencia = "Ligera tendencia ascendente en los valores históricos."
                    else:
                        tendencia = "Ligera tendencia descendente en los valores históricos."

                # Variabilidad (coeficiente de variación)
                mu_safe = mu if mu != 0 else 1e-9
                cv = abs(sigma / mu_safe)

                if cv < 0.10:
                    var_txt = "Baja: los valores se agrupan muy cerca de la media."
                elif cv < 0.25:
                    var_txt = "Media: hay variación normal alrededor de la media."
                else:
                    var_txt = "Alta: la variable presenta mucha dispersión entre muestras."

                # Puntos fuera de control en histórico
                if "ucl" in locals() and "lcl" in locals():
                    fuera_control_hist = (
                        (df_hist["valor"] > ucl) | (df_hist["valor"] < lcl)
                    ).mean()
                else:
                    fuera_control_hist = 0.0

                if fuera_control_hist == 0:
                    riesgo_hist = "Muy bajo: no se observan valores históricos fuera de control estadístico."
                elif fuera_control_hist < 0.05:
                    riesgo_hist = f"Bajo: ~{fuera_control_hist*100:.1f}% de los puntos históricos cae fuera de los límites."
                else:
                    riesgo_hist = f"Alto: ~{fuera_control_hist*100:.1f}% de los puntos históricos supera los límites de control."

                # Riesgo en el forecast
                if not df_pred.empty and "ucl" in locals() and "lcl" in locals():
                    fuera_control_pred = (
                        (df_pred["valor"] > ucl) | (df_pred["valor"] < lcl)
                    ).any()
                    if fuera_control_pred:
                        riesgo_future = (
                            "Hay valores pronosticados que podrían caer fuera de los límites "
                            "de control; se recomienda monitorear de cerca futuros análisis."
                        )
                    else:
                        riesgo_future = (
                            "Los valores pronosticados se mantienen dentro de los límites de control; "
                            "no se esperan cambios bruscos si las condiciones se mantienen."
                        )
                else:
                    riesgo_future = "Sin suficientes datos de pronóstico para evaluar riesgo futuro."

                st.markdown(
                    f"""
**Tendencia histórica:**  
- {tendencia}

**Nivel de variabilidad:**  
- {var_txt}

**Riesgo según historial (carta de control):**  
- {riesgo_hist}

**Comportamiento esperado del forecast:**  
- {riesgo_future}
"""
                )
    else:
        st.info("No existe archivo de forecast.")

# ====================== TAB 4 — VARIABILIDAD ======================
with tab4:
    st.subheader("📊 Variabilidad por Departamento")
    if var_icd_col in df_filtered.columns:
        fig_box = px.box(df_filtered, x=group_col, y=var_icd_col)
        fig_box.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No hay suficientes datos para analizar la variabilidad.")

# ====================== TAB 5 — EXPORT ======================
with tab5:
    st.subheader("⬇ Exportar datos filtrados")
    if var_icd_col in df_filtered.columns:
        cols_export = [c for c in [group_col, "region", "municipio", "cultivo", var_icd_col] if c in df_filtered.columns]
        csv_bytes = df_filtered[cols_export].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar CSV filtrado",
            csv_bytes,
            file_name=f"ICD_{selected_var}_filtrado.csv",
            mime="text/csv",
        )
    else:
        st.info("No hay datos ICD disponibles para exportar con los filtros actuales.")

# ====================== TAB 6 — ANOMALÍAS ======================
with tab6:
    st.subheader("🚨 Anomalías detectadas")

    if "anom_score_global" in df.columns:
        df_anom = df_filtered.sort_values("anom_score_global", ascending=True).head(50)
        fig_anom = px.scatter(
            df_filtered,
            x=selected_var if selected_var in df_filtered.columns else df_filtered.select_dtypes(include=[np.number]).columns[0],
            y="anom_score_global",
            color="anom_score_global",
            color_continuous_scale="Reds",
            hover_data=[c for c in [group_col, "municipio", "cultivo"] if c in df_filtered.columns],
            title=f"Distribución de anomalías — {selected_var}",
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        st.write("🔎 Top 50 registros más sospechosos según el score de anomalía:")
        st.dataframe(df_anom.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No se detectaron anomalías (no existe la columna 'anom_score_global').")

# ====================== TAB 7 — PERFIL AGRONÓMICO + GUÍA ICD ======================
with tab7:
    st.subheader("🧠 Perfil agronómico del suelo (según filtros aplicados)")

    if df_filtered.empty:
        st.info("No hay datos con los filtros actuales. Ajusta región / departamento / cultivo.")
    else:
        st.markdown("**Contexto del perfil:**")
        ctx = []
        if "region" in df.columns and region != "Todas":
            ctx.append(f"Región: `{region}`")
        if depto != "Todos":
            ctx.append(f"Departamento: `{depto}`")
        if muni != "Todos":
            ctx.append(f"Municipio: `{muni}`")
        if cultivo != "Todos":
            ctx.append(f"Cultivo: `{cultivo}`")
        if not ctx:
            ctx.append("Sin filtros específicos (perfil nacional).")
        st.write(" · ".join(ctx))

        st.write("---")
        st.markdown("### 📊 Variables clave del suelo (promedios con filtros)")

        vars_clave = [
            "ph_agua_suelo",
            "materia_organica",
            "fosforo_bray_ii",
            "conductividad_electrica",
            "cic",
        ]
        presentes = [v for v in vars_clave if v in df_filtered.columns]

        if not presentes:
            st.info("No se encontraron variables clave estándar (pH, MO, P, CE, CIC) en el subconjunto filtrado.")
        else:
            resumen = []
            for v in presentes:
                serie = df_filtered[v].dropna()
                if serie.empty:
                    continue
                valor = float(serie.mean())

                if v == "ph_agua_suelo":
                    if valor < 5.5:
                        estado = "Ácido (riesgo de baja disponibilidad de nutrientes)."
                    elif valor > 7.5:
                        estado = "Alcalino (posible bloqueo de micronutrientes)."
                    else:
                        estado = "Óptimo para la mayoría de cultivos."
                elif v == "materia_organica":
                    if valor < 2:
                        estado = "Baja — suelos pobres en materia orgánica; recomendables enmiendas orgánicas."
                    elif valor <= 4:
                        estado = "Media — aceptable, pero podría mejorarse para mayor resiliencia."
                    else:
                        estado = "Alta — buena estructura y reserva de nutrientes."
                elif v == "fosforo_bray_ii":
                    if valor < 15:
                        estado = "Bajo — puede limitar el desarrollo radicular y la producción."
                    elif valor <= 30:
                        estado = "Adecuado para muchos cultivos."
                    else:
                        estado = "Alto — revisar dosis de fertilización fosfatada."
                elif v == "conductividad_electrica":
                    if valor < 2:
                        estado = "Sin problemas importantes de salinidad."
                    elif valor <= 4:
                        estado = "Riesgo moderado de salinidad; algunos cultivos sensibles pueden verse afectados."
                    else:
                        estado = "Alto riesgo de salinidad — revisar calidad de agua y manejo de sales."
                elif v == "cic":
                    if valor < 10:
                        estado = "Baja capacidad de intercambio catiónico — suelos ligeros y con baja retención de nutrientes."
                    elif valor <= 25:
                        estado = "CIC media — capacidad de retención de nutrientes aceptable."
                    else:
                        estado = "CIC alta — buena capacidad para retener nutrientes y amortiguar cambios."
                else:
                    estado = "Sin regla específica, interpretar con un agrónomo."

                resumen.append(
                    {
                        "Variable": v,
                        "Promedio": round(valor, 2),
                        "Interpretación": estado,
                    }
                )

            if resumen:
                st.dataframe(pd.DataFrame(resumen), use_container_width=True)
            else:
                st.info("No hay datos no nulos para las variables clave con los filtros actuales.")

        st.write("---")
        st.markdown("### 📈 ¿Cómo interpretar el ICD en este panel?")
        st.markdown(
            """
**ICD (Índice de Calidad del Dato)** es un valor entre `0` y `1` que combina:

- ✅ Completitud (si las variables tienen datos o muchos vacíos).
- 🚨 Anomalías (valores muy raros o sospechosos).
- 🤖 Coherencia predictiva (si el dato concuerda con lo que espera un modelo de IA).

Rangos usados en ICD Soil:

- `ICD ≥ 0.90` → 🟢 Excelente: datos muy confiables.
- `0.75 ≤ ICD < 0.90` → 🔵 Bueno: datos utilizables con bajo riesgo.
- `0.60 ≤ ICD < 0.75` → 🟡 Moderado: revisar antes de decisiones críticas.
- `0.45 ≤ ICD < 0.60` → 🟠 Alto riesgo: posible error de medición o muestreo.
- `ICD < 0.45` → 🔴 Crítico: no usar sin verificación adicional.

💡 Combina esta guía con el tab **“🧪 Validar muestra”** para revisar análisis recientes de laboratorio.
"""
        )

# ====================== TAB 8 — VALIDACIÓN INTELIGENTE ======================
with tab8:
    st.subheader("🧪 Validación inteligente del análisis")
    st.write("Ingresa valores medidos. Los faltantes se estiman según cultivo y ubicación.")

    variables_ingreso = st.multiselect("Variables a validar:", variables, key="v_ingreso")

    if variables_ingreso:
        valores_usuario = {}

        # Contexto de referencia
        if cultivo != "Todos" and depto != "Todos":
            ref = df[(df[group_col] == depto) & (df["cultivo"] == cultivo)]
        elif depto != "Todos":
            ref = df[df[group_col] == depto]
        else:
            ref = df

        for v in variables_ingreso:
            if v in ref.columns:
                med_ref = ref[v].median()
            else:
                med_ref = df[v].median() if v in df.columns else 0.0

            valores_usuario[v] = st.number_input(
                f"{v} (ref ≈ {round(med_ref, 2)}):",
                value=float(round(med_ref, 2)) if not np.isnan(med_ref) else 0.0,
                key=f"in_{v}",
            )

        if st.button("🔍 Evaluar muestra", key="eval_button"):
            resultados = []
            icd_scores = []

            # 1) Límites físicos de seguridad (ejemplo básico)
            limites_fisicos = {
                "ph_agua_suelo": (0, 14),
                "conductividad_electrica": (0, 20),
                "materia_organica": (0, 40),
            }

            # 2) Mensajes base por nivel
            recomendaciones_base = {
                "Excelente": "✔ Valores dentro del rango esperado. Continuar el manejo actual.",
                "Bueno": "📌 Dato aceptable. Se recomienda hacer seguimiento periódico.",
                "Moderado": "⚠ Puede impactar la productividad según cultivo. Ajustes moderados recomendados.",
                "Alto Riesgo": "🚨 Posible problema agronómico o error analítico. Verificar laboratorio o manejo.",
                "Crítico": "🛑 Fuera de rango técnico. Repetir análisis y aplicar correcciones urgentes.",
            }

            # 3) Reglas específicas por variable
            reglas_variable = {
                "ph_agua_suelo": {
                    "bajo": "Aplicar cal agrícola o dolomita para corregir acidez.",
                    "alto": "Aplicar materia orgánica y evitar fertilizantes alcalinos.",
                },
                "materia_organica": {
                    "bajo": "Incrementar aplicaciones de compost, abonos verdes o enmiendas orgánicas.",
                    "alto": "Evitar excesos que puedan generar problemas de drenaje o enfermedades.",
                },
                "fosforo_bray_ii": {
                    "bajo": "Aplicar fuentes fosfatadas (MAP, DAP o roca fosfórica) según cultivo.",
                    "alto": "Revisar dosis fosfatadas para evitar fijación o pérdidas.",
                },
                "boro_disponible": {
                    "bajo": "Un leve aumento de boro puede mejorar floración y amarre (con mucho cuidado).",
                    "alto": "Riesgo de toxicidad por boro: evitar fertilizantes con B y mejorar drenaje.",
                },
            }

            # 4) Reglas por cultivo (algunos ejemplos)
            reglas_cultivo = {
                "CAFÉ": {
                    "ph_agua_suelo": "Para café se recomiendan pH entre 5.0 y 6.0; fuera de ese rango puede disminuir rendimiento.",
                    "materia_organica": "El café responde muy bien a suelos con MO ≥ 3 %; aprovecha enmiendas orgánicas.",
                },
                "ARROZ": {
                    "conductividad_electrica": "Salinidades altas afectan germinación y macollamiento en arroz.",
                    "fosforo_bray_ii": "El arroz en suelos ácidos suele requerir correcciones fuertes de fósforo.",
                },
                "MAÍZ": {
                    "materia_organica": "Suelos con MO baja requieren buen manejo de residuos y fertilización nitrogenada.",
                    "fosforo_bray_ii": "El maíz es muy sensible a deficiencia de fósforo en etapas tempranas.",
                },
                "PAPA": {
                    "ph_agua_suelo": "La papa prefiere pH ligeramente ácidos (5.0–6.0); pH mayores favorecen sarna.",
                    "conductividad_electrica": "Salinidades altas afectan desarrollo de tubérculos y calidad.",
                },
                "PLÁTANO": {
                    "materia_organica": "El plátano responde a MO alta; los mulches y residuos son aliados claves.",
                    "potasio_intercambiable": "El potasio es crítico en plátano para calidad de racimos y llenado.",
                },
                "PASTO": {
                    "materia_organica": "La ganadería sostenible se apoya en suelos con MO moderada-alta.",
                    "ph_agua_suelo": "pH muy ácidos limitan la oferta forrajera y la fijación de nitrógeno.",
                },
            }

            cultivo_upper = cultivo.upper() if isinstance(cultivo, str) else "TODOS"

            for v in variables_ingreso:
                valor = valores_usuario[v]

                if v in ref.columns:
                    serie = ref[v].dropna()
                elif v in df.columns:
                    serie = df[v].dropna()
                else:
                    serie = pd.Series(dtype=float)

                # Validación física
                if v in limites_fisicos:
                    min_f, max_f = limites_fisicos[v]
                    if valor < min_f or valor > max_f:
                        st.error(
                            f"🚨 {v}: `{valor}` está fuera del rango físico permitido ({min_f}–{max_f}). "
                            "Posible error de laboratorio o de digitación."
                        )
                        continue

                # Estadística (z-score)
                if not serie.empty and serie.std() > 0:
                    z = (valor - serie.mean()) / serie.std()
                else:
                    z = 0.0

                # Nivel según z-score
                if abs(z) < 1:
                    nivel, color, score = "Excelente", "🟢", 0.95
                elif abs(z) < 2:
                    nivel, color, score = "Bueno", "🔵", 0.85
                elif abs(z) < 3:
                    nivel, color, score = "Moderado", "🟡", 0.70
                elif abs(z) < 4:
                    nivel, color, score = "Alto Riesgo", "🟠", 0.50
                else:
                    nivel, color, score = "Crítico", "🔴", 0.25

                icd_scores.append(score)

                # Recomendación base
                recomendacion = recomendaciones_base.get(nivel, "")

                # Ajuste según variable
                if v in reglas_variable:
                    if nivel in ["Moderado", "Alto Riesgo", "Crítico"]:
                        rec_var = reglas_variable[v].get("alto", "")
                    else:
                        rec_var = reglas_variable[v].get("bajo", "")
                    if rec_var:
                        recomendacion += " " + rec_var

                # Ajuste según cultivo
                if cultivo_upper in reglas_cultivo and v in reglas_cultivo[cultivo_upper]:
                    recomendacion += " " + reglas_cultivo[cultivo_upper][v]

                resultados.append(
                    {
                        "variable": v,
                        "valor": valor,
                        "nivel": nivel,
                        "icd": score,
                        "color": color,
                        "z": round(float(z), 2),
                        "recomendacion": recomendacion,
                    }
                )

            # Mostrar resultados
            for r in resultados:
                st.markdown(
                    f"""
---
<div class="ag-card">
<h4>{r['color']} {r['variable']}</h4>
<ul>
    <li><b>Valor ingresado:</b> {r['valor']}</li>
    <li><b>Clasificación:</b> {r['nivel']}</li>
    <li><b>ICD estimado:</b> {round(r['icd'], 2)}</li>
    <li><b>Z-score:</b> {r['z']}</li>
</ul>
<b>Recomendación:</b> {r['recomendacion']}
</div>
""",
                    unsafe_allow_html=True,
                )

            if icd_scores:
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

# ====================== TAB 9 — PCA MULTIVARIABLE ======================
with tab9:
    st.subheader("📉 Análisis por Componentes Principales (PCA)")
    st.write(
        "Este módulo resume varias variables de suelo en 2 componentes principales "
        "para visualizar patrones entre departamentos y cultivos según los filtros aplicados."
    )

    df_pca = df_filtered.copy()
    num_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()

    default_pca = [c for c in num_cols if c.startswith("icd_total_")]
    if not default_pca:
        default_pca = num_cols[:5]

    vars_pca = st.multiselect(
        "Selecciona variables numéricas para el PCA:",
        options=num_cols,
        default=default_pca,
        key="vars_pca",
    )

    if len(vars_pca) < 2:
        st.info("Selecciona al menos 2 variables numéricas para calcular el PCA.")
    else:
        df_pca_clean = df_pca.dropna(subset=vars_pca).copy()

        if df_pca_clean.shape[0] < 5:
            st.warning("No hay suficientes registros sin valores faltantes para hacer PCA con los filtros actuales.")
        else:
            X = df_pca_clean[vars_pca].values.astype(float)

            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_std[X_std == 0] = 1.0
            X_scaled = (X - X_mean) / X_std

            pca = PCA(n_components=2, random_state=42)
            comps = pca.fit_transform(X_scaled)

            df_pca_clean["PC1"] = comps[:, 0]
            df_pca_clean["PC2"] = comps[:, 1]

            exp_var = pca.explained_variance_ratio_
            st.markdown(
                f"**Varianza explicada:** PC1 = `{exp_var[0]*100:.1f}%`, "
                f"PC2 = `{exp_var[1]*100:.1f}%` (total ≈ `{(exp_var[0]+exp_var[1])*100:.1f}%`)."
            )

            color_col = None
            if "cultivo" in df_pca_clean.columns and cultivo == "Todos":
                color_col = "cultivo"
            elif group_col in df_pca_clean.columns:
                color_col = group_col
            elif "region" in df_pca_clean.columns:
                color_col = "region"

            st.write("---")
            st.markdown("### 🌐 Mapa de muestras en el espacio PCA")

            hover_cols = []
            if group_col in df_pca_clean.columns:
                hover_cols.append(group_col)
            if "cultivo" in df_pca_clean.columns:
                hover_cols.append("cultivo")

            fig_pca = px.scatter(
                df_pca_clean,
                x="PC1",
                y="PC2",
                color=color_col,
                hover_data=hover_cols if hover_cols else None,
                title="Distribución de registros en componentes principales",
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            st.write("---")
            st.markdown("### 🧬 Contribución de las variables a PC1 y PC2")

            loadings = pd.DataFrame(
                pca.components_.T, columns=["PC1", "PC2"], index=vars_pca
            ).reset_index().rename(columns={"index": "Variable"})

            fig_load = px.bar(
                loadings.melt(
                    id_vars="Variable",
                    value_vars=["PC1", "PC2"],
                    var_name="Componente",
                    value_name="Carga",
                ),
                x="Variable",
                y="Carga",
                color="Componente",
                barmode="group",
                title="Cargas de cada variable en PC1 y PC2",
            )
            fig_load.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_load, use_container_width=True)

            st.info(
                "💡 Variables con cargas altas (positivas o negativas) en una componente "
                "son las que más influyen en esa dirección del espacio multivariable."
            )

# ====================== FOOTER ======================
st.markdown(
    "<div class='ag-footer'>✔ ICD Soil — Panel integrado de calidad de dato, anomalías, forecast, validación y PCA.</div>",
    unsafe_allow_html=True,
)
