"""
generar_datos_experimento_salud_mental.py
------------------------------------------
Variante experimental de generar_datos.py para un ejercicio pedido por el
profesor: ver cómo cambia la importancia de la variable
"problemas_salud_mental" en los modelos si esta variable tiene una señal
más fuerte y realista.

Cambios respecto a generar_datos.py (única y exclusivamente en este script):
    1. problemas_salud_mental: p=[0.38, 0.62] en vez de p=[0.18, 0.82]
       (más casos positivos, más balanceado).
    2. En la fórmula de z, el coeficiente de problemas_salud_mental sube
       de 1.1 a 2.3 (efecto mucho más determinante y claro).

El resto de variables y relaciones son idénticas a generar_datos.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 77
N_ESTUDIANTES = 400

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "experimento_salud_mental.csv"


def generar_dataset_experimento(n=N_ESTUDIANTES, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    edad = rng.integers(17, 30, size=n)
    genero = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
    semestre_actual = rng.integers(1, 11, size=n)
    estrato_socioeconomico = rng.choice([1, 2, 3, 4, 5, 6], size=n,
                                         p=[0.18, 0.30, 0.27, 0.15, 0.07, 0.03])
    promedio_academico = np.clip(rng.normal(3.6, 0.6, size=n), 1.0, 5.0)
    porcentaje_asistencia = np.clip(rng.normal(82, 14, size=n), 20, 100)
    materias_perdidas_semestre = np.clip(
        rng.poisson(lam=1.1, size=n) + (5 - promedio_academico).clip(min=0).astype(int) // 2,
        0, 8
    )
    horas_trabajo_semanal = np.clip(rng.exponential(scale=8, size=n), 0, 50).astype(int)
    distancia_campus_km = np.clip(rng.exponential(scale=6, size=n), 0.5, 60)
    nivel_educativo_padres = rng.choice(
        ["Ninguno", "Primaria", "Secundaria", "Técnico", "Universitario", "Posgrado"],
        size=n, p=[0.05, 0.15, 0.30, 0.20, 0.22, 0.08]
    )
    apoyo_financiero = rng.choice(["Si", "No"], size=n, p=[0.35, 0.65])
    beca = rng.choice(["Si", "No"], size=n, p=[0.22, 0.78])
    dificultades_economicas = rng.choice(["Si", "No"], size=n, p=[0.40, 0.60])
    participacion_actividades = rng.choice(["Si", "No"], size=n, p=[0.30, 0.70])
    # Cambio 1: señal de salud mental más fuerte y balanceada
    problemas_salud_mental = rng.choice(["Si", "No"], size=n, p=[0.38, 0.62])

    nivel_padres_score = pd.Series(nivel_educativo_padres).map({
        "Ninguno": 0, "Primaria": 1, "Secundaria": 2,
        "Técnico": 3, "Universitario": 4, "Posgrado": 5
    }).to_numpy()

    def z_score(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    z = (
        -1.6 * z_score(porcentaje_asistencia)
        - 1.9 * z_score(promedio_academico)
        + 1.3 * z_score(materias_perdidas_semestre)
        + 0.7 * z_score(horas_trabajo_semanal)
        + 0.3 * z_score(distancia_campus_km)
        - 0.8 * z_score(nivel_padres_score)
        + 1.4 * (dificultades_economicas == "Si")
        + 2.3 * (problemas_salud_mental == "Si")  # Cambio 2: 1.1 -> 2.3
        - 1.0 * (apoyo_financiero == "Si")
        - 0.9 * (beca == "Si")
        - 0.5 * (participacion_actividades == "Si")
        - 0.15 * z_score(semestre_actual)
        - 1.35
    )
    ruido = rng.normal(0, 0.9, size=n)
    prob_abandono = 1 / (1 + np.exp(-(z + ruido)))
    prob_abandono = np.clip(prob_abandono, 0.02, 0.97)
    abandono = rng.binomial(1, prob_abandono)

    df = pd.DataFrame({
        "edad": edad,
        "genero": genero,
        "semestre_actual": semestre_actual,
        "estrato_socioeconomico": estrato_socioeconomico,
        "promedio_academico": promedio_academico.round(2),
        "porcentaje_asistencia": porcentaje_asistencia.round(1),
        "materias_perdidas_semestre": materias_perdidas_semestre,
        "horas_trabajo_semanal": horas_trabajo_semanal,
        "distancia_campus_km": distancia_campus_km.round(1),
        "nivel_educativo_padres": nivel_educativo_padres,
        "apoyo_financiero": apoyo_financiero,
        "beca": beca,
        "dificultades_economicas": dificultades_economicas,
        "participacion_actividades": participacion_actividades,
        "problemas_salud_mental": problemas_salud_mental,
        "abandono": abandono,
    })

    return df


if __name__ == "__main__":
    df = generar_dataset_experimento()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Dataset del experimento generado: {len(df)} estudiantes")
    print(f"Tasa de abandono simulada: {df['abandono'].mean():.1%}")
    print(f"Tasa de problemas_salud_mental='Si': {(df['problemas_salud_mental'] == 'Si').mean():.1%}")
    print(f"Guardado en: {OUTPUT_PATH}")
