from pathlib import Path
import pandas as pd

# =========================
# RUTAS BASE DEL PROYECTO
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

# GeoJSON oficial ya existente en tu proyecto
GEOJSON_PATH = BASE_DIR / "data" / "geojson" / "colombia_departamentos.json"

# Lista estándar de variables de suelo (opcional)
SOIL_VARIABLES = [
    "ph_agua_suelo", "materia_organica", "fosforo_bray_ii",
    "azufre_fosfato_monocalcico", "acidez_intercambiable",
    "aluminio_intercambiable", "calcio_intercambiable",
    "magnesio_intercambiable", "potasio_intercambiable",
    "sodio_intercambiable", "cic", "conductividad_electrica",
    "hierro_olsen", "cobre_disponible", "manganeso_olsen",
    "zinc_olsen", "boro_disponible", "hierro_doble_acido",
    "cobre_disponible_doble_acido", "manganeso_doble_acido",
    "zinc_doble_acido",
]

# =========================
# CARGA PRINCIPAL DE DATOS
# =========================
def load_data():
    """
    Carga los datasets procesados principales para la app:

    - suelos_icd_registro.csv            → df
    - suelos_icd_depto_variable.csv      → df_depto
    - forecast_global_por_variable.csv   → df_forecast (opcional)

    Devuelve: df, df_depto, df_forecast
    """

    icd_registro_path = PROCESSED_DIR / "suelos_icd_registro.csv"
    icd_depto_var_path = PROCESSED_DIR / "suelos_icd_depto_variable.csv"

    if not icd_registro_path.exists():
        raise FileNotFoundError(
            f"No se encontró {icd_registro_path}. "
            "Ejecuta el notebook 06_ICD_por_variable para generarlo."
        )

    if not icd_depto_var_path.exists():
        raise FileNotFoundError(
            f"No se encontró {icd_depto_var_path}. "
            "Ejecuta el notebook 06_ICD_por_variable para generarlo."
        )

    # Carga principal
    df = pd.read_csv(icd_registro_path)
    df_depto = pd.read_csv(icd_depto_var_path)

    # Asegurar que las columnas ICD_total sean numéricas
    icd_total_cols = [c for c in df.columns if c.startswith("icd_total_")]
    df[icd_total_cols] = df[icd_total_cols].apply(pd.to_numeric, errors="coerce")

    # Forecast (opcional)
    forecast_path = PROCESSED_DIR / "forecast_global_por_variable.csv"
    if forecast_path.exists():
        df_forecast = pd.read_csv(forecast_path)
        if "fecha" in df_forecast.columns:
            df_forecast["fecha"] = pd.to_datetime(df_forecast["fecha"], errors="coerce")
    else:
        df_forecast = None

    return df, df_depto, df_forecast


# =========================
# UTILIDADES PARA LA APP
# =========================
def get_variables(df: pd.DataFrame):
    """
    Devuelve la lista de variables de suelo disponibles a partir
    de las columnas 'icd_total_<variable>' presentes en df.
    """
    if df is None:
        return []

    vars_cols = [c for c in df.columns if c.startswith("icd_total_")]
    variables = sorted(c.replace("icd_total_", "") for c in vars_cols)
    return variables


def get_group_col(df: pd.DataFrame) -> str:
    """
    Devuelve el nombre de la columna territorial principal.
    """
    if df is None:
        return ""

    if "dep_norm" in df.columns:
        return "dep_norm"
    elif "departamento" in df.columns:
        return "departamento"
    elif "DEPARTAMENTO" in df.columns:
        return "DEPARTAMENTO"
    else:
        return ""
