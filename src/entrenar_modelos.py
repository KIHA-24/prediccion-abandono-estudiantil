"""
entrenar_modelos.py
--------------------
Entrena y compara tres modelos de clasificación supervisada para predecir
el abandono estudiantil, tal como sugiere el proyecto:

    1. Regresión Logística (Logistic Regression)
    2. Random Forest
    3. XGBoost

Flujo:
    1. Carga el dataset (data/estudiantes.csv)
    2. Preprocesa: codifica variables categóricas, escala variables numéricas
    3. Divide en entrenamiento/prueba (80/20, estratificado)
    4. Entrena los 3 modelos
    5. Evalúa con Accuracy, Precisión, Recall, F1 y ROC-AUC
    6. Guarda el mejor modelo (según F1) + el preprocesador en /models
    7. Genera una tabla comparativa y gráficas en /reports

Ejecutar desde la raíz del proyecto:
    python src/entrenar_modelos.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "estudiantes.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

COLUMNAS_NUMERICAS = [
    "edad", "semestre_actual", "estrato_socioeconomico", "promedio_academico",
    "porcentaje_asistencia", "materias_perdidas_semestre", "horas_trabajo_semanal",
    "distancia_campus_km",
]
COLUMNAS_CATEGORICAS = [
    "genero", "nivel_educativo_padres", "apoyo_financiero", "beca",
    "dificultades_economicas", "participacion_actividades", "problemas_salud_mental",
]
COLUMNA_OBJETIVO = "abandono"


def cargar_datos() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta primero: python src/generar_datos.py"
        )
    return pd.read_csv(DATA_PATH)


def construir_preprocesador() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), COLUMNAS_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
    ])


def evaluar_modelo(nombre, modelo, X_test, y_test, resultados, y_score_metodo="predict_proba"):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metrics = {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    resultados.append(metrics)

    print(f"\n{'=' * 55}\n{nombre}\n{'=' * 55}")
    print(classification_report(y_test, y_pred, target_names=["Continúa", "Abandono"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    return y_pred, y_proba


def graficar_matrices_confusion(matrices, REPORTS_DIR):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.2))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (nombre, cm) in zip(axes, matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Continúa", "Abandono"],
                    yticklabels=["Continúa", "Abandono"])
        ax.set_title(nombre)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "matrices_confusion.png", dpi=150)
    plt.close()


def graficar_comparacion(df_resultados, REPORTS_DIR):
    df_plot = df_resultados.set_index("modelo")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ax = df_plot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Comparación de modelos - Predicción de abandono estudiantil")
    ax.set_ylabel("Puntaje")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "comparacion_modelos.png", dpi=150)
    plt.close()


def obtener_importancias(modelo):
    """
    Devuelve feature_importances_ del modelo. Si viene envuelto en un
    CalibratedClassifierCV (que no expone feature_importances_ directamente),
    promedia el atributo entre los estimadores base ajustados en cada fold
    interno de la calibración (calibrated_classifiers_[i].estimator en
    scikit-learn >= 1.0; algunas versiones anteriores usan base_estimator).
    Devuelve None si no se puede obtener por ningún medio (p. ej. modelos
    lineales como LogisticRegression, que no tienen feature_importances_).
    """
    if hasattr(modelo, "feature_importances_"):
        return modelo.feature_importances_

    calibrados = getattr(modelo, "calibrated_classifiers_", None)
    if not calibrados:
        return None

    importancias_por_fold = []
    for cc in calibrados:
        estimador_base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if estimador_base is not None and hasattr(estimador_base, "feature_importances_"):
            importancias_por_fold.append(estimador_base.feature_importances_)

    if not importancias_por_fold:
        return None
    return np.mean(importancias_por_fold, axis=0)


def obtener_coeficientes(modelo):
    """
    Devuelve los coeficientes (coef_) de un modelo lineal binario, promediando
    entre los estimadores base ajustados en cada fold interno de la calibración
    si el modelo viene envuelto en un CalibratedClassifierCV (mismo mecanismo que
    obtener_importancias). Devuelve None si el modelo no expone coef_ por ningún
    medio (p. ej. modelos basados en árboles como Random Forest o XGBoost).
    """
    if hasattr(modelo, "coef_"):
        return modelo.coef_[0]

    calibrados = getattr(modelo, "calibrated_classifiers_", None)
    if not calibrados:
        return None

    coefs_por_fold = []
    for cc in calibrados:
        estimador_base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if estimador_base is not None and hasattr(estimador_base, "coef_"):
            coefs_por_fold.append(estimador_base.coef_[0])

    if not coefs_por_fold:
        return None
    return np.mean(coefs_por_fold, axis=0)


def graficar_coeficientes_regresion_logistica(modelo, nombre_columnas, REPORTS_DIR, nombre_modelo):
    coeficientes = obtener_coeficientes(modelo)
    if coeficientes is None:
        print(
            f"[Aviso] No se genera gráfica de coeficientes para '{nombre_modelo}': "
            f"el modelo no expone coef_ (ni directamente ni a través de "
            f"calibrated_classifiers_) — normal para modelos basados en árboles."
        )
        return

    orden = np.argsort(np.abs(coeficientes))[::-1][:15]
    nombres = np.array(nombre_columnas)[orden]
    valores = coeficientes[orden]
    colores = ["#D62728" if v > 0 else "#4C72B0" for v in valores]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=valores, y=nombres, hue=nombres, palette=colores, legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Coeficientes - {nombre_modelo}\n(rojo = aumenta el riesgo, azul = lo reduce)")
    plt.xlabel("Coeficiente (escala estandarizada)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "coeficientes_regresion_logistica.png", dpi=150)
    plt.close()
    print(f"Gráfica de coeficientes generada para '{nombre_modelo}'.")


def graficar_importancia_variables(modelo, nombre_columnas, REPORTS_DIR, nombre_modelo):
    importancias = obtener_importancias(modelo)
    if importancias is None:
        print(
            f"[Aviso] No se genera gráfica de importancia de variables para "
            f"'{nombre_modelo}': el modelo no expone feature_importances_ "
            f"(ni directamente ni a través de calibrated_classifiers_)."
        )
        return
    orden = np.argsort(importancias)[::-1][:15]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importancias[orden], y=np.array(nombre_columnas)[orden], color="#4C72B0")
    plt.title(f"Variables más importantes - {nombre_modelo}")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"importancia_variables_{nombre_modelo.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"Gráfica de importancia de variables generada para '{nombre_modelo}'.")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Cargando datos...")
    df = cargar_datos()
    X = df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS]
    y = df[COLUMNA_OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    preprocesador = construir_preprocesador()

    modelos = {
        "Regresión Logística": CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            method="sigmoid", cv=5,
        ),
        "Random Forest": CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            method="sigmoid", cv=5,
        ),
    }
    if XGBOOST_DISPONIBLE:
        modelos["XGBoost"] = CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                eval_metric="logloss", random_state=42,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            ),
            method="sigmoid", cv=5,
        )
    else:
        print("\n[Aviso] xgboost no está instalado en este entorno. "
              "Instálalo con 'pip install xgboost' para incluirlo en la comparación.\n")

    resultados = []
    matrices_confusion = {}
    pipelines_entrenados = {}

    for nombre, modelo in modelos.items():
        pipeline = Pipeline(steps=[
            ("preprocesador", preprocesador),
            ("clasificador", modelo),
        ])
        pipeline.fit(X_train, y_train)
        pipelines_entrenados[nombre] = pipeline

        y_pred, y_proba = evaluar_modelo(nombre, pipeline, X_test, y_test, resultados)
        matrices_confusion[nombre] = confusion_matrix(y_test, y_pred)

    df_resultados = pd.DataFrame(resultados).sort_values("f1", ascending=False)
    print(f"\n{'=' * 55}\nTABLA COMPARATIVA\n{'=' * 55}")
    print(df_resultados.round(3).to_string(index=False))

    df_resultados.round(4).to_csv(REPORTS_DIR / "tabla_comparativa.csv", index=False)

    # Gráficas
    graficar_matrices_confusion(matrices_confusion, REPORTS_DIR)
    graficar_comparacion(df_resultados, REPORTS_DIR)

    # Curvas ROC
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    for nombre, pipeline in pipelines_entrenados.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=nombre, ax=ax)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Curvas ROC")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "curvas_roc.png", dpi=150)
    plt.close()

    # Mejor modelo -> guardar
    mejor_nombre = df_resultados.iloc[0]["modelo"]
    mejor_pipeline = pipelines_entrenados[mejor_nombre]
    joblib.dump(mejor_pipeline, MODELS_DIR / "mejor_modelo.joblib")

    # Guardar también todos, por si se quieren comparar en la app
    # (nombres de archivo fijos para que app.py los reconozca)
    ARCHIVO_POR_MODELO = {
        "Regresión Logística": "regresión_logística.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }
    for nombre, pipeline in pipelines_entrenados.items():
        joblib.dump(pipeline, MODELS_DIR / ARCHIVO_POR_MODELO[nombre])

    # Importancia de variables (para modelos basados en árboles)
    nombres_columnas_transformadas = (
        COLUMNAS_NUMERICAS +
        list(mejor_pipeline.named_steps["preprocesador"]
             .named_transformers_["cat"].get_feature_names_out(COLUMNAS_CATEGORICAS))
    )
    for nombre, pipeline in pipelines_entrenados.items():
        clasificador = pipeline.named_steps["clasificador"]
        graficar_importancia_variables(clasificador, nombres_columnas_transformadas, REPORTS_DIR, nombre)
        graficar_coeficientes_regresion_logistica(clasificador, nombres_columnas_transformadas, REPORTS_DIR, nombre)

    with open(MODELS_DIR / "info_modelo.json", "w", encoding="utf-8") as f:
        json.dump({
            "mejor_modelo": mejor_nombre,
            "metricas": df_resultados[df_resultados["modelo"] == mejor_nombre].iloc[0].to_dict(),
            "columnas_numericas": COLUMNAS_NUMERICAS,
            "columnas_categoricas": COLUMNAS_CATEGORICAS,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nMejor modelo: {mejor_nombre} (guardado en models/mejor_modelo.joblib)")
    print(f"Reportes y gráficas guardados en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
