import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from PIL import Image
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

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

    df['precipitacion_lag1'] = df['precipitacion_valor'].shift(1)
    df['temperatura_lag1'] = df['temperatura_valor'].shift(1)
    df['nivel_agua_lag1'] = df['nivel_agua_valor'].shift(1)
    df['presion_lag1'] = df['presion_valor'].shift(1)
    df = df.dropna()

    features = ['Mes', 'Año', 'precipitacion_lag1', 'temperatura_lag1', 'nivel_agua_lag1', 'presion_lag1']

    modelos = {
        'rf': {
            'precipitacion': RandomForestRegressor(n_estimators=100, random_state=42),
            'nivel_agua': RandomForestRegressor(n_estimators=100, random_state=42)
        },
        'arima': {
            'precipitacion': ARIMA(df['precipitacion_valor'], order=(3, 1, 2)).fit(),
            'nivel_agua': ARIMA(df['nivel_agua_valor'], order=(2, 1, 2)).fit()
        }
    }

    modelos['rf']['precipitacion'].fit(df[features], df['precipitacion_valor'])
    modelos['rf']['nivel_agua'].fit(df[features], df['nivel_agua_valor'])

    return modelos, df

# ===============================
# FUNCIÓN DE PRECISIÓN TEMPORAL
# ===============================
def calcular_precision_temporal(fecha_objetivo, fecha_ultima):
    dias_diff = (fecha_objetivo - fecha_ultima).days
    if dias_diff <= 0:
        return 1.0
    max_dias = 365 * 2
    precision = max(0.3, 1 - (dias_diff / max_dias))
    return round(precision * 100, 1)

# ===============================
# PREDICCIÓN A FUTURO CON FUSIÓN
# ===============================
def predecir_variables(modelos, mes, año, dia, ultimos_valores, df):
    fecha_pred = pd.Timestamp(f"{año}-{mes}-{dia}")
    fecha_ultima = df['fecha'].max()
    dias_diff = (fecha_pred - fecha_ultima).days
    alpha = min(1.0, dias_diff / 180)

    entrada = [[
        mes,
        año,
        ultimos_valores['precipitacion_valor'],
        ultimos_valores['temperatura_valor'],
        ultimos_valores['nivel_agua_valor'],
        ultimos_valores['presion_valor']
    ]]

    pred_rf = {
        'precipitacion': modelos['rf']['precipitacion'].predict(entrada)[0],
        'nivel_agua': modelos['rf']['nivel_agua'].predict(entrada)[0]
    }

    pasos = dias_diff if dias_diff > 0 else 1
    pred_arima = {
        'precipitacion': modelos['arima']['precipitacion'].forecast(steps=pasos).iloc[-1],
        'nivel_agua': modelos['arima']['nivel_agua'].forecast(steps=pasos).iloc[-1]
    }

    pred_fusion = {
        k: round(max(0, alpha * pred_arima[k] + (1 - alpha) * pred_rf[k]), 2)
        for k in ['precipitacion', 'nivel_agua']
    }

    precision = calcular_precision_temporal(fecha_pred, fecha_ultima)
    fechas_precision = pd.date_range(start=fecha_ultima + pd.Timedelta(days=1), end=fecha_pred, freq='D')
    valores_precision = [calcular_precision_temporal(f, fecha_ultima) for f in fechas_precision]

    return pred_fusion, precision, fechas_precision, valores_precision

# ===============================
# GRÁFICAS Y GEMINI
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

    interpretacion = interpretar_variable(variable, pred_valores[-1])
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return Image.open(buf), interpretacion

def crear_grafica_precision(fechas, precisiones):
    plt.figure(figsize=(8, 4))
    plt.plot(fechas, precisiones, marker='o', color='green')
    plt.ylim(0, 105)
    plt.title("📉 Precisión esperada del modelo en el tiempo")
    plt.ylabel("Precisión estimada (%)")
    plt.xlabel("Fecha de predicción")
    plt.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

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
        f"en La Gasca, Quito. Da una recomendación práctica para la comunidad."
    )
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error con Gemini API - {variable.capitalize()}]: {str(e)}"