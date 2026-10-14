"""
predecir.py
-----------
Ejemplo rápido de cómo usar el modelo entrenado para predecir el riesgo
de abandono de UN estudiante, sin necesidad de abrir la interfaz web.
Útil para probar el modelo desde la terminal o integrarlo en otro sistema.

Ejecutar:
    python src/predecir.py
"""

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_PATH = BASE_DIR / "models" / "mejor_modelo.joblib"


def predecir_estudiante(datos_estudiante: dict) -> dict:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Ejecuta primero:\n"
            "  python src/generar_datos.py\n"
            "  python src/entrenar_modelos.py"
        )
    modelo = joblib.load(MODELO_PATH)
    df = pd.DataFrame([datos_estudiante])
    probabilidad = modelo.predict_proba(df)[0, 1]
    prediccion = int(modelo.predict(df)[0])
    return {"prediccion": prediccion, "probabilidad_abandono": round(float(probabilidad), 4)}


if __name__ == "__main__":
    # Ejemplo: estudiante con varias señales de riesgo
    estudiante_ejemplo = {
        "edad": 19,
        "genero": "M",
        "semestre_actual": 2,
        "estrato_socioeconomico": 2,
        "promedio_academico": 2.8,
        "porcentaje_asistencia": 60,
        "materias_perdidas_semestre": 3,
        "horas_trabajo_semanal": 25,
        "distancia_campus_km": 18,
        "nivel_educativo_padres": "Primaria",
        "apoyo_financiero": "No",
        "beca": "No",
        "dificultades_economicas": "Si",
        "participacion_actividades": "No",
        "problemas_salud_mental": "No",
    }

    resultado = predecir_estudiante(estudiante_ejemplo)
    print("Datos del estudiante de ejemplo:")
    for k, v in estudiante_ejemplo.items():
        print(f"  {k}: {v}")
    print("\nResultado de la predicción:")
    print(f"  ¿Riesgo de abandono?: {'SI' if resultado['prediccion'] == 1 else 'NO'}")
    print(f"  Probabilidad de abandono: {resultado['probabilidad_abandono']:.1%}")
