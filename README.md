# 🎓 Predicción de Abandono Estudiantil

**Proyecto N.º 12** · Área: Aprendizaje Supervisado (Clasificación)
Prototipo desarrollado como propuesta de investigación académica.

---

## 1. Planteamiento del problema

La **deserción estudiantil** (dropout) es uno de los problemas más estudiados
en la gestión académica universitaria. Un estudiante que abandona representa
una pérdida para sí mismo, para su familia y para la institución (recursos,
cupos, indicadores de calidad ante el Ministerio de Educación Nacional y
acreditación).

**Pregunta de investigación:**
¿Es posible, a partir de variables académicas, socioeconómicas y de
comportamiento disponibles tempranamente, predecir qué estudiantes tienen
mayor riesgo de abandonar sus estudios, para intervenir a tiempo?

## 2. Marco teórico (para sustentar ante el profesor)

- **Modelo de Integración Social de Vincent Tinto (1975, 1993):** el abandono
  ocurre cuando falla la integración académica y social del estudiante con
  la institución.
- **Modelo de Spady (1970):** la deserción es un proceso multicausal
  (rendimiento académico + factores sociales + compromiso institucional).
- **Estudios del MEN (Colombia) y SPADIES:** identifican como variables de
  alerta temprana el rendimiento académico, la situación socioeconómica
  (estrato, apoyo financiero), la carga laboral del estudiante y el nivel
  educativo de los padres — todas incluidas en este prototipo.

Estas referencias te sirven para justificar **por qué** se eligieron las
variables del modelo (no son arbitrarias, están basadas en literatura).

## 3. Objetivo del prototipo

Construir y comparar tres modelos de **clasificación supervisada** —
**Regresión Logística, Random Forest y XGBoost** — que, a partir del perfil
de un estudiante, predigan la probabilidad de que abandone sus estudios, y
presentarlo mediante una interfaz interactiva que simule una herramienta de
alerta temprana para Bienestar Universitario / Registro Académico.

## 4. Metodología (CRISP-DM aplicado)

| Fase | Qué se hizo |
|---|---|
| Comprensión del negocio | Definición del problema y variables relevantes (sección 1-2) |
| Comprensión y preparación de los datos | Dataset sintético con relaciones realistas (`src/generar_datos.py`) — ver nota abajo |
| Modelado | 3 algoritmos supervisados de clasificación binaria |
| Evaluación | Accuracy, Precisión, Recall, F1-score y ROC-AUC (train/test 80/20 estratificado) |
| Despliegue | Prototipo web interactivo con Streamlit (`app.py`) |

> **Nota importante sobre los datos:** por no contar con acceso a datos reales
> y confidenciales de estudiantes de una institución, este prototipo usa un
> **dataset sintético** generado de forma programática, pero con relaciones
> lógicas y realistas entre variables (a menor asistencia y promedio, y a
> mayor dificultad económica, mayor probabilidad de abandono), tal como
> describe la literatura citada. Esto es habitual y válido en la fase de
> **prototipo/prueba de concepto**. En una fase posterior, este mismo
> pipeline se puede reentrenar con datos reales del Registro Académico de la
> universidad (bajo los debidos permisos y anonimización, conforme a la
> Ley 1581 de 2012 de Protección de Datos Personales en Colombia).

## 5. Variables utilizadas

| Variable | Tipo | Descripción |
|---|---|---|
| edad, genero, semestre_actual | Demográfica | Datos básicos del estudiante |
| estrato_socioeconomico | Socioeconómica | Estrato 1-6 |
| promedio_academico | Académica | Promedio acumulado (escala 0.0-5.0) |
| porcentaje_asistencia | Académica | % de asistencia a clases |
| materias_perdidas_semestre | Académica | N.º de materias perdidas en el semestre actual |
| horas_trabajo_semanal | Socioeconómica | Horas que trabaja a la semana |
| distancia_campus_km | Contextual | Distancia de la vivienda al campus |
| nivel_educativo_padres | Socioeconómica | Nivel educativo alcanzado por los padres |
| apoyo_financiero, beca | Socioeconómica | Si cuenta con apoyo económico institucional |
| dificultades_economicas | Socioeconómica | Autoreporte de dificultades económicas |
| participacion_actividades | Comportamiento | Participación en actividades extracurriculares |
| problemas_salud_mental | Bienestar | Autoreporte de dificultades emocionales/salud mental |
| **abandono** (objetivo) | — | 1 = desertó, 0 = continúa activo |

## 6. Estructura del proyecto

```
prediccion-abandono-estudiantil/
├── data/
│   └── estudiantes.csv          # Dataset generado (se crea al ejecutar el paso 1)
├── models/                      # Modelos entrenados (.joblib) (se crean en el paso 2)
├── reports/                     # Gráficas y tabla comparativa (se crean en el paso 2)
├── src/
│   ├── generar_datos.py         # Paso 1: genera el dataset sintético
│   ├── entrenar_modelos.py      # Paso 2: entrena y compara los 3 modelos
│   └── predecir.py              # Ejemplo de predicción por consola
├── app.py                       # Interfaz web interactiva (Streamlit)
├── requirements.txt
└── README.md
```

## 7. Cómo ejecutar el proyecto en VS Code

### Paso 0 — Requisitos
- Tener Python 3.10+ instalado.
- Abrir la carpeta `prediccion-abandono-estudiantil` en VS Code.
- Instalar la extensión oficial **Python** de Microsoft (si no la tienes).

### Paso 1 — Crear entorno virtual e instalar dependencias

Abre una terminal en VS Code (`Ctrl + ñ` o `Terminal > New Terminal`) y ejecuta:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

En VS Code, selecciona ese entorno como intérprete: `Ctrl+Shift+P` →
`Python: Select Interpreter` → elige el que dice `.venv`.

### Paso 2 — Generar el dataset

```bash
python src/generar_datos.py
```

Esto crea `data/estudiantes.csv` con 1500 estudiantes simulados.

### Paso 3 — Entrenar y comparar los modelos

```bash
python src/entrenar_modelos.py
```

Esto entrena los 3 modelos, imprime las métricas en consola, y guarda:
- `models/*.joblib` (los 3 modelos + el mejor por separado)
- `reports/*.png` (matrices de confusión, curvas ROC, importancia de variables)
- `reports/tabla_comparativa.csv`

### Paso 4 — Lanzar la aplicación interactiva (para la demo)

```bash
streamlit run app.py
```

Se abrirá automáticamente en el navegador (normalmente en
`http://localhost:8501`). Ahí podrás:
1. Llenar el formulario con el perfil de un estudiante y predecir su riesgo.
2. Ver la comparación completa entre los 3 modelos (pestaña "Comparación de modelos").
3. Explorar el dataset usado (pestaña "Dataset").

**Tip para la sustentación:** prueba con un estudiante de bajo promedio,
baja asistencia y dificultades económicas (riesgo alto) y luego con uno de
buen promedio y apoyo financiero (riesgo bajo), para mostrar cómo cambia la
predicción en vivo.

## 8. Resultados esperados (orientativos)

Con el dataset sintético incluido, los 3 modelos alcanzan métricas
razonables para un prototipo (ROC-AUC aproximado entre 0.80 y 0.90), y
normalmente **XGBoost o Random Forest** superan levemente a la Regresión
Logística en F1-score, aunque la Regresión Logística es más fácil de
interpretar (coeficientes). Los valores exactos se generan al ejecutar
`entrenar_modelos.py` y pueden variar levemente por la semilla aleatoria.

## 9. Próximos pasos (para la sustentación / trabajo futuro)

- Sustituir el dataset sintético por datos reales anonimizados de la institución.
- Ajuste de hiperparámetros (GridSearchCV / Optuna) para mejorar el rendimiento.
- Validación cruzada (k-fold) para una evaluación más robusta.
- Explicabilidad del modelo con SHAP, para justificar cada predicción individual.
- Integrar el modelo a un sistema real de alerta temprana conectado al
  sistema de información académico de la universidad.
- Monitoreo de *data drift*: reentrenar periódicamente con datos nuevos.

## 10. Tecnologías utilizadas

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn ·
Streamlit · Joblib

## 11. Carga incremental de datos por semestre

Para no tener que regenerar el dataset completo cada vez que hay un nuevo
semestre de estudiantes, la app incluye una pestaña **"📤 Cargar nuevo
semestre"** que permite acumular datos progresivamente y reentrenar los
modelos con la información más reciente, sin salir de la interfaz.

### Cómo funciona

1. **Subir el CSV**: en esa pestaña, se sube un archivo `.csv` con exactamente
   las mismas columnas del dataset actual (las variables predictoras de la
   sección 5, más la columna `abandono`). El sistema valida las columnas antes
   de aceptarlo (`gestion_datos.validar_csv_nuevo`); si falta o sobra alguna,
   muestra un error indicando cuál y no deja continuar.
2. **Vista previa**: si el CSV es válido, se muestra una vista previa de las
   primeras filas para confirmar visualmente que los datos se ven correctos.
3. **Confirmar y anexar**: al presionar el botón, las filas nuevas se
   **anexan** a `data/estudiantes.csv` (`gestion_datos.anexar_datos`) — el
   dataset existente nunca se borra ni se sobrescribe, solo crece. Cada carga
   queda registrada en `data/historial_cargas.csv`
   (`gestion_datos.registrar_carga`), con fecha, nombre del archivo, cuántos
   estudiantes se agregaron y el total acumulado — esto da trazabilidad de
   qué se cargó y cuándo.
4. **Reentrenar**: un botón separado ejecuta de nuevo `entrenar_modelos.py`
   (reutilizando su misma lógica, sin duplicar código) sobre el dataset ya
   acumulado, actualiza `models/*.joblib` y `reports/tabla_comparativa.csv`, y
   limpia la caché de Streamlit para que el resto de la app (predicción
   individual, comparación de modelos) use los modelos nuevos de inmediato,
   sin reiniciar el servidor.

### Limitación conocida

Por ahora el sistema **asume que cada carga trae estudiantes nuevos**: no
existe una columna de identificador único de estudiante, así que no puede
detectar ni manejar el caso de que un mismo estudiante aparezca en más de una
carga (por ejemplo, para actualizar su información de un semestre a otro). Si
se sube dos veces el mismo CSV por error, el estudiante quedará duplicado en
el dataset.

**Trabajo futuro:** agregar una columna de ID único de estudiante y una
lógica de *upsert* (actualizar la fila si el ID ya existe, insertar si es
nuevo) en `anexar_datos`, en vez de la concatenación simple actual.
