"""
gestion_datos.py
-----------------
Funciones para acumular datos de nuevos semestres al dataset existente
(data/estudiantes.csv) y llevar un historial de cargas (data/historial_cargas.csv).

Las columnas esperadas se reutilizan directamente de entrenar_modelos.py para
no duplicar esa definición.

Ejecutar como script:
    (no aplica — este módulo se usa desde app.py)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from entrenar_modelos import (
    COLUMNAS_NUMERICAS,
    COLUMNAS_CATEGORICAS,
    COLUMNA_OBJETIVO,
    DATA_PATH,
)

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORIAL_PATH = BASE_DIR / "data" / "historial_cargas.csv"

COLUMNAS_ESPERADAS = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + [COLUMNA_OBJETIVO]

COLUMNAS_HISTORIAL = [
    "fecha_carga", "nombre_archivo",
    "n_estudiantes_nuevos", "n_estudiantes_total_acumulado",
]


def validar_csv_nuevo(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Verifica que df tenga exactamente las columnas que espera el modelo y que
    no tenga celdas vacías (NaN) en ninguna de ellas. No intenta imputar ni
    corregir nada — si hay celdas vacías, rechaza el archivo con un mensaje
    que dice en qué columnas y cuántas filas, para que el usuario las
    complete antes de volver a cargarlo.
    """
    columnas_df = set(df.columns)
    columnas_esperadas = set(COLUMNAS_ESPERADAS)

    faltantes = sorted(columnas_esperadas - columnas_df)
    sobrantes = sorted(columnas_df - columnas_esperadas)

    if faltantes or sobrantes:
        partes = []
        if faltantes:
            partes.append(f"faltan las columnas {faltantes}")
        if sobrantes:
            partes.append(f"sobran columnas no reconocidas {sobrantes}")
        mensaje = "El CSV no coincide con las columnas del modelo: " + "; ".join(partes) + "."
        return False, mensaje

    nulos_por_columna = df[COLUMNAS_ESPERADAS].isna().sum()
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    if not columnas_con_nulos.empty:
        detalle = ", ".join(
            f"{col} ({n} fila{'s' if n != 1 else ''})"
            for col, n in columnas_con_nulos.items()
        )
        mensaje = f"El archivo tiene celdas vacías en: {detalle}. Complétalas antes de cargar el archivo."
        return False, mensaje

    return True, ""


def normalizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los valores de texto en las columnas categóricas (strip + una
    capitalización consistente) para que variantes como "m"/"M", "no"/"NO"/" No"
    no se conviertan en categorías distintas para el OneHotEncoder, diluyendo
    la señal de esa variable. No toca columnas numéricas ni "abandono".

    Se agregó después de detectar el problema con data/prueba_excel_realista.xlsx:
    el validador (validar_csv_nuevo) solo revisa que existan las columnas
    correctas, no el contenido de los valores dentro de ellas.
    """
    df = df.copy()
    for col in COLUMNAS_CATEGORICAS:
        if col not in df.columns:
            continue
        no_nulos = df[col].notna()
        valores = df.loc[no_nulos, col].astype(str).str.strip()
        if col == "genero":
            df.loc[no_nulos, col] = valores.str.upper()
        else:
            df.loc[no_nulos, col] = valores.str.capitalize()
    return df


def anexar_datos(df_nuevo: pd.DataFrame) -> int:
    """Concatena df_nuevo a data/estudiantes.csv (lo crea si no existe) y devuelve el total de filas."""
    df_nuevo = df_nuevo[COLUMNAS_ESPERADAS]
    df_nuevo = normalizar_texto(df_nuevo)

    if DATA_PATH.exists():
        df_actual = pd.read_csv(DATA_PATH)
        df_final = pd.concat([df_actual, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(DATA_PATH, index=False, encoding="utf-8")
    return len(df_final)


def registrar_carga(nombre_archivo: str, n_filas_nuevas: int, n_filas_total: int) -> None:
    """Agrega una fila de trazabilidad a data/historial_cargas.csv (lo crea si no existe)."""
    fila = pd.DataFrame([{
        "fecha_carga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nombre_archivo": nombre_archivo,
        "n_estudiantes_nuevos": n_filas_nuevas,
        "n_estudiantes_total_acumulado": n_filas_total,
    }])

    if HISTORIAL_PATH.exists():
        historial = pd.concat([pd.read_csv(HISTORIAL_PATH), fila], ignore_index=True)
    else:
        historial = fila

    HISTORIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    historial.to_csv(HISTORIAL_PATH, index=False, encoding="utf-8")


def obtener_historial_cargas() -> pd.DataFrame:
    """Devuelve el historial de cargas, o un DataFrame vacío si todavía no existe."""
    if not HISTORIAL_PATH.exists():
        return pd.DataFrame(columns=COLUMNAS_HISTORIAL)
    return pd.read_csv(HISTORIAL_PATH)
