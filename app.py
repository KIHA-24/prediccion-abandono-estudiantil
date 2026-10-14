"""
app.py
------
Aplicación web interactiva (Streamlit) para demostrar el prototipo de
"Predicción de abandono estudiantil" al profesor.

Permite:
    - Ingresar los datos de un estudiante mediante un formulario
    - Obtener la predicción (Riesgo / No riesgo) y la probabilidad
    - Ver qué modelo se está usando y comparar los 3 modelos entrenados
    - Ver estadísticas generales del dataset y gráficas de resultados

Ejecutar (desde la raíz del proyecto, con el entorno virtual activado):
    streamlit run app.py
"""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_PATH = BASE_DIR / "data" / "estudiantes.csv"

sys.path.insert(0, str(BASE_DIR / "src"))
import entrenar_modelos  # noqa: E402
import gestion_datos  # noqa: E402

st.set_page_config(
    page_title="Predicción de Abandono Estudiantil",
    page_icon="🎓",
    layout="wide",
)

NOMBRES_MODELOS_DISPONIBLES = {
    "Regresión Logística": "regresión_logística.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


@st.cache_resource
def cargar_modelo(archivo: str):
    ruta = MODELS_DIR / archivo
    if not ruta.exists():
        return None
    return joblib.load(ruta)


@st.cache_data
def cargar_dataset(mtime: float | None):
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def formatear_probabilidad(proba: float) -> str:
    if proba > 0.999:
        return ">99.9%"
    if proba < 0.001:
        return "<0.1%"
    return f"{proba:.1%}"


@st.cache_data
def cargar_info_modelo():
    ruta = MODELS_DIR / "info_modelo.json"
    if not ruta.exists():
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.title("🎓 Predicción de Abandono Estudiantil")
    st.caption(
        "Prototipo académico — Proyecto: *Predicción de abandono estudiantil* · "
        "Área: Aprendizaje supervisado · Modelos: Regresión Logística, Random Forest, XGBoost"
    )

    info_modelo = cargar_info_modelo()
    if info_modelo is None:
        st.error(
            "No se encontraron modelos entrenados. Ejecuta primero, en una terminal:\n\n"
            "1. `python src/generar_datos.py`\n"
            "2. `python src/entrenar_modelos.py`"
        )
        st.stop()

    tab_prediccion, tab_comparacion, tab_datos, tab_carga = st.tabs(
        ["🔮 Predicción individual", "📊 Comparación de modelos", "📁 Dataset", "📤 Cargar nuevo semestre"]
    )

    # ---------------------------------------------------------------
    # TAB 1: Predicción individual
    # ---------------------------------------------------------------
    with tab_prediccion:
        col_form, col_resultado = st.columns([1.3, 1])

        with col_form:
            st.subheader("Datos del estudiante")

            modelos_disponibles = [
                nombre for nombre, archivo in NOMBRES_MODELOS_DISPONIBLES.items()
                if (MODELS_DIR / archivo).exists()
            ]
            modelo_elegido = st.selectbox(
                "Modelo a utilizar",
                modelos_disponibles,
                index=modelos_disponibles.index(info_modelo["mejor_modelo"])
                if info_modelo["mejor_modelo"] in modelos_disponibles else 0,
                help="Por defecto se recomienda el modelo con mejor F1-score en las pruebas.",
            )

            c1, c2 = st.columns(2)
            with c1:
                edad = st.number_input("Edad", 16, 45, 20)
                genero = st.selectbox("Género", ["M", "F"])
                semestre_actual = st.slider("Semestre actual", 1, 10, 3)
                estrato_socioeconomico = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6], index=1)
                promedio_academico = st.slider("Promedio académico (0.0 - 5.0)", 0.0, 5.0, 3.5, 0.1)
                porcentaje_asistencia = st.slider("Porcentaje de asistencia (%)", 0, 100, 85)
                materias_perdidas_semestre = st.number_input("Materias perdidas este semestre", 0, 10, 0)
            with c2:
                horas_trabajo_semanal = st.number_input("Horas de trabajo semanales", 0, 60, 0)
                distancia_campus_km = st.number_input("Distancia al campus (km)", 0.0, 100.0, 5.0)
                nivel_educativo_padres = st.selectbox(
                    "Nivel educativo de los padres",
                    ["Ninguno", "Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"],
                    index=2,
                )
                apoyo_financiero = st.selectbox("¿Recibe apoyo financiero?", ["Si", "No"], index=1)
                beca = st.selectbox("¿Tiene beca?", ["Si", "No"], index=1)
                dificultades_economicas = st.selectbox("¿Dificultades económicas?", ["Si", "No"], index=1)
                participacion_actividades = st.selectbox("¿Participa en actividades extracurriculares?", ["Si", "No"], index=1)
            problemas_salud_mental = st.selectbox("¿Reporta problemas de salud mental/emocional?", ["Si", "No"], index=1)

            predecir_btn = st.button("Predecir riesgo de abandono", type="primary", use_container_width=True)

        with col_resultado:
            st.subheader("Resultado")
            if predecir_btn:
                modelo = cargar_modelo(NOMBRES_MODELOS_DISPONIBLES[modelo_elegido])
                entrada = pd.DataFrame([{
                    "edad": edad,
                    "genero": genero,
                    "semestre_actual": semestre_actual,
                    "estrato_socioeconomico": estrato_socioeconomico,
                    "promedio_academico": promedio_academico,
                    "porcentaje_asistencia": porcentaje_asistencia,
                    "materias_perdidas_semestre": materias_perdidas_semestre,
                    "horas_trabajo_semanal": horas_trabajo_semanal,
                    "distancia_campus_km": distancia_campus_km,
                    "nivel_educativo_padres": nivel_educativo_padres,
                    "apoyo_financiero": apoyo_financiero,
                    "beca": beca,
                    "dificultades_economicas": dificultades_economicas,
                    "participacion_actividades": participacion_actividades,
                    "problemas_salud_mental": problemas_salud_mental,
                }])

                proba = modelo.predict_proba(entrada)[0, 1]
                prediccion = modelo.predict(entrada)[0]

                proba_texto = formatear_probabilidad(proba)
                if prediccion == 1:
                    st.error(f"⚠️ RIESGO DE ABANDONO — Probabilidad: {proba_texto}")
                else:
                    st.success(f"✅ Sin riesgo relevante — Probabilidad de abandono: {proba_texto}")

                st.progress(min(max(proba, 0.0), 1.0))

                if proba >= 0.7:
                    nivel = "Alto 🔴"
                elif proba >= 0.4:
                    nivel = "Medio 🟠"
                else:
                    nivel = "Bajo 🟢"
                st.metric("Nivel de riesgo", nivel)

                st.caption(f"Modelo usado: **{modelo_elegido}**")

                with st.expander("Recomendaciones sugeridas para Bienestar Universitario"):
                    if proba >= 0.4:
                        st.markdown(
                            "- Programar tutoría académica y seguimiento personalizado.\n"
                            "- Evaluar remisión a Bienestar Universitario (apoyo psicosocial/financiero).\n"
                            "- Revisar posibilidad de becas o auxilios económicos.\n"
                            "- Hacer seguimiento cercano de asistencia en las próximas semanas."
                        )
                    else:
                        st.markdown("- Continuar seguimiento académico habitual.")
            else:
                st.info("Completa el formulario y presiona **Predecir riesgo de abandono**.")

    # ---------------------------------------------------------------
    # TAB 2: Comparación de modelos
    # ---------------------------------------------------------------
    with tab_comparacion:
        st.subheader("Comparación de los 3 modelos entrenados")
        tabla_path = REPORTS_DIR / "tabla_comparativa.csv"
        if tabla_path.exists():
            df_tabla = pd.read_csv(tabla_path)
            st.dataframe(df_tabla, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            img = REPORTS_DIR / "comparacion_modelos.png"
            if img.exists():
                st.image(str(img), caption="Métricas por modelo")
        with col_b:
            img = REPORTS_DIR / "curvas_roc.png"
            if img.exists():
                st.image(str(img), caption="Curvas ROC")

        img = REPORTS_DIR / "matrices_confusion.png"
        if img.exists():
            st.image(str(img), caption="Matrices de confusión")

        st.markdown("#### Variables más importantes")
        for archivo in REPORTS_DIR.glob("importancia_variables_*.png"):
            st.image(str(archivo))

        img = REPORTS_DIR / "coeficientes_regresion_logistica.png"
        if img.exists():
            st.markdown("#### Coeficientes de la Regresión Logística")
            st.image(str(img))

    # ---------------------------------------------------------------
    # TAB 3: Dataset
    # ---------------------------------------------------------------
    with tab_datos:
        st.subheader("Dataset utilizado (sintético)")
        mtime_actual = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else None
        df = cargar_dataset(mtime_actual)
        if df is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total de estudiantes", len(df))
            c2.metric("Tasa de abandono", f"{df['abandono'].mean():.1%}")
            c3.metric("Variables predictoras", len(df.columns) - 1)
            st.dataframe(df.head(50), use_container_width=True)
            st.caption(
                "Nota: este es un dataset SINTÉTICO generado para fines de prototipo. "
                "En una versión de producción se reemplazaría por datos reales de la "
                "institución, garantizando el tratamiento adecuado de datos personales."
            )

    # ---------------------------------------------------------------
    # TAB 4: Cargar nuevo semestre
    # ---------------------------------------------------------------
    with tab_carga:
        st.subheader("Acumular un nuevo semestre de datos")
        st.caption(
            "Sube un CSV o Excel (.xlsx) con exactamente las mismas columnas del dataset actual "
            "(incluyendo la columna 'abandono') para anexarlo y, si quieres, reentrenar los modelos."
        )

        archivo_subido = st.file_uploader("Archivo CSV o Excel del nuevo semestre", type=["csv", "xlsx"])

        if archivo_subido is not None:
            try:
                if archivo_subido.name.lower().endswith(".xlsx"):
                    df_nuevo = pd.read_excel(archivo_subido)
                else:
                    df_nuevo = pd.read_csv(archivo_subido)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
                df_nuevo = None

            if df_nuevo is not None:
                es_valido, mensaje_error = gestion_datos.validar_csv_nuevo(df_nuevo)
                if not es_valido:
                    st.error(mensaje_error)
                else:
                    st.success(f"Archivo válido — {len(df_nuevo)} estudiantes detectados.")
                    st.markdown("**Vista previa:**")
                    st.dataframe(df_nuevo.head(), use_container_width=True)

                    if st.button("Confirmar y anexar al dataset", type="primary"):
                        n_total = gestion_datos.anexar_datos(df_nuevo)
                        gestion_datos.registrar_carga(archivo_subido.name, len(df_nuevo), n_total)
                        cargar_dataset.clear()
                        st.success(
                            f"Se agregaron {len(df_nuevo)} estudiantes nuevos. "
                            f"Total acumulado: {n_total} estudiantes."
                        )

        st.divider()
        st.subheader("Reentrenar modelos")
        st.caption("Vuelve a entrenar los 3 modelos con todos los datos acumulados hasta ahora.")

        if st.button("🔄 Reentrenar los 3 modelos con los datos acumulados"):
            with st.spinner("Entrenando los 3 modelos con los datos acumulados... esto puede tardar un momento."):
                entrenar_modelos.main()
            cargar_modelo.clear()
            cargar_dataset.clear()
            cargar_info_modelo.clear()
            st.success("Modelos reentrenados y guardados correctamente.")
            st.markdown("**Tabla comparativa actualizada:**")
            tabla_actualizada = pd.read_csv(REPORTS_DIR / "tabla_comparativa.csv")
            st.dataframe(tabla_actualizada, use_container_width=True)

        st.divider()
        st.subheader("Historial de cargas")
        historial = gestion_datos.obtener_historial_cargas()
        if historial.empty:
            st.info("Todavía no se ha cargado ningún semestre adicional.")
        else:
            st.dataframe(historial, use_container_width=True)


if __name__ == "__main__":
    main()
