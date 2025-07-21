import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ======================
# CONFIGURACIÓN
# ======================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CSV_PATH = "dataset_rumipamba_limpio.csv"

# ===============================
# ENTRENAMIENTO DEL MODELO
# ===============================
def entrenar_modelo():
    df = pd.read_csv(CSV_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['Mes'] = df['fecha'].dt.month
    df['Año'] = df['fecha'].dt.year

    # Variables rezagadas
    df['precipitacion_lag1'] = df['precipitacion_valor'].shift(1)
    df['temperatura_lag1'] = df['temperatura_valor'].shift(1)
    df['nivel_agua_lag1'] = df['nivel_agua_valor'].shift(1)
    df['presion_lag1'] = df['presion_valor'].shift(1)
    df = df.dropna()

    features = ['Mes', 'Año', 'precipitacion_lag1', 'temperatura_lag1', 'nivel_agua_lag1', 'presion_lag1']

    modelos = {
        'precipitacion': RandomForestRegressor(n_estimators=100, random_state=42),
        'nivel_agua': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    modelos['precipitacion'].fit(df[features], df['precipitacion_valor'])
    modelos['nivel_agua'].fit(df[features], df['nivel_agua_valor'])

    return modelos, df

# ===============================
# PREDICCIÓN A FUTURO
# ===============================
def predecir_variables(modelos, mes, año, ultimos_valores):
    entrada = [[
        mes,
        año,
        ultimos_valores['precipitacion_valor'],
        ultimos_valores['temperatura_valor'],
        ultimos_valores['nivel_agua_valor'],
        ultimos_valores['presion_valor']
    ]]

    pred = {
        'precipitacion': modelos['precipitacion'].predict(entrada)[0],
        'nivel_agua': modelos['nivel_agua'].predict(entrada)[0]
    }

    pred = {k: max(0, round(v, 2)) for k, v in pred.items()}
    return pred

# ===============================
# CREACIÓN DE GRÁFICA VISUAL
# ===============================
def crear_grafica(modelo, df, mes, año, variable, color='blue'):
    plt.figure(figsize=(12, 5))
    fechas = df['fecha']
    valores = df[f"{variable}_valor"]
    plt.plot(fechas, valores, label=f"{variable.capitalize()} histórica", marker='o', color=color)

    X_all = df[['Mes', 'Año', 'precipitacion_lag1', 'temperatura_lag1', 'nivel_agua_lag1', 'presion_lag1']]
    y_pred = modelo.predict(X_all)
    plt.plot(fechas, y_pred, linestyle='--', label="Tendencia estimada", color='orange')

    ult = df.sort_values(by='fecha').iloc[-1]
    fecha_inicio = ult['fecha']
    fecha_fin = pd.to_datetime(f"{año}-{mes}-01")
    fechas_futuras = pd.date_range(start=fecha_inicio + pd.DateOffset(months=1), end=fecha_fin, freq='MS')

    pred_fechas, pred_valores = [], []
    lag_precipitacion = ult['precipitacion_valor']
    lag_temperatura = ult['temperatura_valor']
    lag_nivel_agua = ult['nivel_agua_valor']
    lag_presion = ult['presion_valor']

    for f in fechas_futuras:
        entrada = [[f.month, f.year, lag_precipitacion, lag_temperatura, lag_nivel_agua, lag_presion]]
        pred = modelo.predict(entrada)[0]
        ruido = np.random.normal(loc=0, scale=pred * 0.15)
        pred_ajustada = max(0, pred + ruido)
        pred_fechas.append(f)
        pred_valores.append(pred_ajustada)

        # Actualizar valores rezagados
        if variable == 'precipitacion':
            lag_precipitacion = pred_ajustada
        elif variable == 'nivel_agua':
            lag_nivel_agua = pred_ajustada

    if pred_fechas:
        plt.plot(pred_fechas, pred_valores, linestyle='-', color='gray', label="Proyección mensual futura")
        plt.scatter(pred_fechas[-1], pred_valores[-1], color='red', s=100, label=f"Predicción para {mes}/{año}")

    plt.title(f"{variable.capitalize()} mensual y predicción")
    plt.xlabel("Fecha")
    plt.ylabel(variable.capitalize())
    plt.grid(True)
    plt.legend(loc='upper left')

    # Interpretación separada
    interpretacion = interpretar_variable(variable, pred_valores[-1])

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return Image.open(buf), interpretacion

# ===============================
# INTERPRETACIÓN CON GEMINI
# ===============================
def interpretar_con_gemini(pred):
    precip = pred['precipitacion']
    nivel = pred['nivel_agua']

    prompt = (
        f"Actúa como experto en prevención de desbordamientos urbanos. Interpreta los siguientes datos climáticos "
        f"para la zona de La Gasca, Quito:\n\n"
        f"1. Precipitación estimada: {precip} mm\n"
        f"2. Nivel de agua estimado: {nivel} cm\n\n"
        f"¿Qué riesgo representa esto y qué recomendaciones deberíamos dar a la comunidad?"
    )

    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error con Gemini API]: {str(e)}"

def interpretar_variable(variable, valor):
    prompt = (
        f"Como experto en clima urbano, explica en máximo 2 líneas qué significa un valor de {valor} para {variable} "
        f"en La Gasca, Quito. Da una recomendación práctica para la comunidad (prevención, drenaje, evacuación, etc.)."
    )
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error con Gemini API - {variable.capitalize()}]: {str(e)}"