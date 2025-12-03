import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px

from utils import load_data, get_variables, get_group_col, GEOJSON_PATH

# ====================== CONFIG ======================
st.set_page_config(page_title="AgroGuardian ICD", layout="wide")

# ====================== LOAD DATA =====================
df, df_depto, df_forecast = load_data()

group_col = get_group_col(df) or "departamento"
variables = get_variables(df)

# ====================== HEADER ======================
st.title("🌱 AgroGuardian — Índice de Calidad del Dato (ICD)")
st.caption("Plataforma institucional | Concurso Datos al Ecosistema 2025")

# ====================== SIDEBAR ======================
st.sidebar.header("🎛 Filtros del análisis")

# ---- Variable de suelo ----
selected_var = st.sidebar.selectbox("📌 Variable de suelo", variables)
var_icd_col = f"icd_total_{selected_var}"

# ---- Región (si existe) ----
if "region" in df.columns:
    regiones = sorted(df["region"].dropna().unique())
    selected_region = st.sidebar.selectbox("🌎 Región", ["Todas"] + regiones)
else:
    selected_region = "Todas"

# ---- Filtro progresivo: región → departamento → municipio ----
df_filter = df.copy()

if selected_region != "Todas" and "region" in df_filter.columns:
    df_filter = df_filter[df_filter["region"] == selected_region]

# Departamentos disponibles
deptos = sorted(df_filter[group_col].dropna().unique())
selected_depto = st.sidebar.selectbox("🏛 Departamento", ["Todos"] + list(deptos))

if selected_depto != "Todos":
    df_filter = df_filter[df_filter[group_col] == selected_depto]

# Municipios (si existe columna)
if "municipio" in df_filter.columns:
    munis = sorted(df_filter["municipio"].dropna().unique())
    selected_muni = st.sidebar.selectbox("📍 Municipio", ["Todos"] + list(munis))
    if selected_muni != "Todos":
        df_filter = df_filter[df_filter["municipio"] == selected_muni]
else:
    selected_muni = "Todos"

df_view = df_filter.copy()

# ====================== ICD INTERPRETATION ======================
def interpretar_icd(valor: float) -> str:
    if pd.isna(valor):
        return "⚫ Sin datos"
    if valor >= 0.95:
        return "🟢 Excelente — Alta confiabilidad"
    elif valor >= 0.85:
        return "🟡 Bueno — Uso recomendado"
    elif valor >= 0.70:
        return "🟠 Regular — Validar antes de usar"
    else:
        return "🔴 Deficiente — Alto riesgo de error"

# ====================== KPI PRINCIPAL ======================
st.subheader(f"📌 Resultados para: **{selected_var}**")

if var_icd_col in df_view.columns:
    avg_icd = df_view[var_icd_col].mean()
    st.metric(label="ICD promedio con filtros aplicados", value=f"{avg_icd:.3f}")
    st.info(f"📖 Evaluación: **{interpretar_icd(avg_icd)}**")
else:
    st.error(f"No se encontró la columna {var_icd_col} en los datos.")
    avg_icd = float("nan")

# ====================== RANKING BASE ======================
if var_icd_col in df_view.columns:
    ranking = (
        df_view.groupby(group_col)[var_icd_col]
        .mean()
        .reset_index()
        .dropna(subset=[var_icd_col])
        .sort_values(var_icd_col, ascending=False)
    )
    ranking["Estado"] = ranking[var_icd_col].apply(interpretar_icd)
else:
    ranking = pd.DataFrame(columns=[group_col, var_icd_col, "Estado"])

# ====================== TABS PRINCIPALES ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Ranking", "🗺 Mapa", "📈 Forecast", "📉 Variabilidad", "⬇ Descargas", "🧪 Simulador"]
)

# ---------------------- TAB 1: RANKING ----------------------
with tab1:
    st.subheader("🏅 Ranking por departamento")

    if ranking.empty:
        st.info("No hay datos suficientes para construir el ranking con los filtros actuales.")
    else:
        modo_movil = st.toggle("📱 Modo compacto", value=False)

        fig_rank = px.bar(
            ranking,
            x=var_icd_col,
            y=group_col,
            text=var_icd_col,
            color="Estado",
            orientation="h",
            title=f"Ranking ICD — {selected_var}",
        )

        fig_rank.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_rank.update_layout(height=400 if modo_movil else 650, yaxis_title="Departamento")

        st.plotly_chart(fig_rank, use_container_width=True)
        st.dataframe(ranking, use_container_width=True)

# ---------------------- TAB 2: MAPA ----------------------
with tab2:
    st.subheader("🗺 Mapa interactivo por departamento")

    if ranking.empty:
        st.info("No hay datos suficientes para mapear la variable seleccionada.")
    else:
        try:
            gdf = gpd.read_file(GEOJSON_PATH)
        except Exception as e:
            st.error(f"No se pudo leer el GeoJSON: {e}")
        else:
            # Normalizar nombres para el join
            def clean_name(s):
                if pd.isna(s):
                    return ""
                return (
                    str(s)
                    .upper()
                    .strip()
                    .encode("ascii", "ignore")
                    .decode("utf-8")
                )

            df_clean = ranking.copy()
            df_clean["clean_dep"] = df_clean[group_col].apply(clean_name)

            geo_key = None
            for col in gdf.columns:
                if col == "geometry":
                    continue
                gdf["_clean_geo"] = gdf[col].apply(clean_name)
                if set(df_clean["clean_dep"]).intersection(set(gdf["_clean_geo"])):
                    geo_key = col
                    break

            if geo_key is None:
                st.error("No se pudo emparejar los nombres de departamentos entre el GeoJSON y el dataset.")
            else:
                merged = gdf.merge(
                    df_clean,
                    left_on="_clean_geo",
                    right_on="clean_dep",
                    how="left",
                )

                estilo = st.selectbox(
                    "🎨 Estilo del mapa",
                    ["Claro", "Satélite", "Oscuro"],
                    index=0,
                )

                map_style = {
                    "Claro": "carto-positron",
                    "Satélite": "satellite-streets",
                    "Oscuro": "carto-darkmatter",
                }[estilo]

                fig_map = px.choropleth_mapbox(
                    merged,
                    geojson=merged.geometry.__geo_interface__,
                    locations=merged.index,
                    color=var_icd_col,
                    hover_name=geo_key,
                    hover_data={var_icd_col: ":.3f", group_col: True},
                    mapbox_style=map_style,
                    zoom=4.5,
                    opacity=0.85,
                    color_continuous_scale="Viridis",
                )

                fig_map.update_layout(
                    height=650,
                    margin=dict(l=0, r=0, t=0, b=0),
                )
                st.plotly_chart(fig_map, use_container_width=True)

# ---------------------- TAB 3: FORECAST ----------------------
with tab3:
    st.subheader("📈 Tendencia y pronóstico")

    if df_forecast is None:
        st.info("⚠ No se encontró archivo de forecast. Ejecuta el notebook 07_forecasting_por_variable.")
    else:
        mask_var = df_forecast["variable"] == selected_var
        df_f = df_forecast[mask_var].copy()

        if df_f.empty:
            st.info("No hay datos de forecast para la variable seleccionada.")
        else:
            # Asegurar tipos de fecha
            df_f["fecha"] = pd.to_datetime(df_f["fecha"], errors="coerce")
            df_f = df_f.dropna(subset=["fecha"])

            fig_forecast = px.line(
                df_f,
                x="fecha",
                y="valor",
                color="tipo",
                markers=True,
                title=f"Serie histórica y pronóstico — {selected_var}",
            )
            fig_forecast.update_layout(xaxis_title="Fecha", yaxis_title=selected_var)
            st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------------- TAB 4: VARIABILIDAD ----------------------
with tab4:
    st.subheader("📉 Variabilidad entre departamentos")

    if ranking.empty:
        st.info("No hay datos suficientes para analizar variabilidad.")
    else:
        fig_box = px.box(
            ranking,
            y=var_icd_col,
            x=group_col,
            points="all",
            title=f"Distribución del ICD por departamento — {selected_var}",
        )
        fig_box.update_layout(
            xaxis_tickangle=45,
            yaxis_title="ICD",
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ---------------------- TAB 5: DESCARGAS ----------------------
with tab5:
    st.subheader("⬇ Exportaciones")

    if ranking.empty:
        st.info("No hay tabla de ranking para descargar.")
    else:
        st.download_button(
            "📥 Descargar ranking (CSV)",
            ranking.to_csv(index=False),
            file_name=f"ICD_ranking_{selected_var}.csv",
            mime="text/csv",
        )

    # Exportar resumen global por variable (si existe en processed/exportables)
    export_folder = PROCESSED_DIR / "exportables"
    resumen_var_path = export_folder / "resumen_icd_por_variable.csv"
    resumen_depto_path = export_folder / "resumen_icd_por_departamento.csv"

    if resumen_var_path.exists():
        with open(resumen_var_path, "rb") as f:
            st.download_button(
                "📥 Descargar resumen ICD por variable",
                f,
                file_name="resumen_icd_por_variable.csv",
                mime="text/csv",
            )

    if resumen_depto_path.exists():
        with open(resumen_depto_path, "rb") as f:
            st.download_button(
                "📥 Descargar resumen ICD por departamento",
                f,
                file_name="resumen_icd_por_departamento.csv",
                mime="text/csv",
            )

# ---------------------- TAB 6: SIMULADOR ----------------------
with tab6:
    st.subheader("🧪 Simulador de Calidad del Dato")
    st.write(
        "Ingresa un valor medido para la variable seleccionada y "
        "evalúa qué tan coherente es respecto al comportamiento histórico."
    )

    if selected_var not in df.columns:
        st.warning(
            f"La variable original '{selected_var}' no está disponible en el dataset de registros. "
            "El simulador requiere los valores originales, no solo el ICD."
        )
    else:
        # Opciones de departamento
        deptos_sim = sorted(df[group_col].dropna().unique())
        colA, colB = st.columns(2)

        departamento_sim = colA.selectbox("Departamento", deptos_sim)
        valor_medido = colB.number_input(
            f"Valor medido de {selected_var}",
            value=float(df[selected_var].dropna().median()) if df[selected_var].notna().any() else 0.0,
        )

        if st.button("🚀 Evaluar muestra"):
            # ---------- Componente nacional ----------
            serie_nat = pd.to_numeric(df[selected_var], errors="coerce")
            nat_mean = serie_nat.mean()
            nat_std = serie_nat.std()

            if pd.isna(nat_std) or nat_std == 0:
                score_z = 0.5
            else:
                z_score = abs((valor_medido - nat_mean) / nat_std)
                score_z = max(0.0, 1.0 - min(z_score / 3.0, 1.0))

            # ---------- Componente regional ----------
            serie_reg = pd.to_numeric(
                df.loc[df[group_col] == departamento_sim, selected_var],
                errors="coerce",
            )
            reg_mean = serie_reg.mean()

            if serie_reg.dropna().empty or pd.isna(reg_mean):
                score_local = 0.5
            else:
                diff_loc = abs(valor_medido - reg_mean)
                # normalizamos por 3 desviaciones estándar regionales si existen
                reg_std = serie_reg.std()
                if pd.isna(reg_std) or reg_std == 0:
                    score_local = max(0.0, 1.0 - min(diff_loc / (abs(reg_mean) + 1e-6), 1.0))
                else:
                    score_local = max(0.0, 1.0 - min(diff_loc / (3 * reg_std), 1.0))

            # ---------- Componente de modelo / forecast ----------
            if df_forecast is not None and selected_var in df_forecast["variable"].unique():
                df_f_var = df_forecast[df_forecast["variable"] == selected_var]
                # usamos solo los valores de forecast
                serie_ml = df_f_var.loc[df_f_var["tipo"] == "forecast", "valor"]
                if serie_ml.dropna().empty:
                    serie_ml = df_f_var["valor"]
                pred_ml = serie_ml.mean()
                diff_ml = abs(valor_medido - pred_ml)
                # Escala suave
                score_ml = max(0.0, 1.0 - min(diff_ml / (3 * serie_ml.std(ddof=0) if serie_ml.std(ddof=0) > 0 else abs(pred_ml) + 1e-6), 1.0))
            else:
                score_ml = 0.6  # valor por defecto si no hay modelo

            # ---------- ICD simulado ----------
            icd_sim = round(0.40 * score_z + 0.30 * score_local + 0.30 * score_ml, 3)
            estado_sim = interpretar_icd(icd_sim)

            col1, col2 = st.columns(2)
            col1.metric("ICD estimado", f"{icd_sim:.3f}")
            col2.metric("Evaluación", estado_sim)

            st.write("---")
            st.subheader("🧠 Recomendación automática")

            if icd_sim >= 0.95:
                st.success("Excelente ✔️ — La muestra es altamente confiable.")
            elif icd_sim >= 0.85:
                st.info("Buena 😉 — Puede usarse sin restricciones, aunque se recomienda monitorear la variabilidad.")
            elif icd_sim >= 0.70:
                st.warning("Regular ⚠️ — Úsala con precaución; considera validar con una muestra adicional.")
            else:
                st.error("Deficiente ❌ — La muestra no es confiable. Se sugiere repetir muestreo o revisar el proceso analítico.")

# Mensaje final
st.success("Panel AgroGuardian ICD listo ✓")
