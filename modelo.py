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
import plotly.graph_objects as go

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
            'nivel_agua': RandomForestRegressor(n_estimators=100, random_state=42),
            'temperatura': RandomForestRegressor(n_estimators=100, random_state=42),
            'presion': RandomForestRegressor(n_estimators=100, random_state=42)
        },
        'arima': {
            'precipitacion': ARIMA(df['precipitacion_valor'], order=(3, 1, 2)).fit(),
            'nivel_agua': ARIMA(df['nivel_agua_valor'], order=(2, 1, 2)).fit(),
            'temperatura': ARIMA(df['temperatura_valor'], order=(2, 1, 2)).fit(),
            'presion': ARIMA(df['presion_valor'], order=(2, 1, 2)).fit()
        }
    }

    # Entrenar modelos Random Forest
    modelos['rf']['precipitacion'].fit(df[features], df['precipitacion_valor'])
    modelos['rf']['nivel_agua'].fit(df[features], df['nivel_agua_valor'])
    modelos['rf']['temperatura'].fit(df[features], df['temperatura_valor'])
    modelos['rf']['presion'].fit(df[features], df['presion_valor'])

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

    # Predicción Random Forest
    pred_rf = {
        'precipitacion': modelos['rf']['precipitacion'].predict(entrada)[0],
        'nivel_agua': modelos['rf']['nivel_agua'].predict(entrada)[0],
        'temperatura': modelos['rf']['temperatura'].predict(entrada)[0],
        'presion': modelos['rf']['presion'].predict(entrada)[0]
    }

    # Predicción ARIMA (solo para variables con series temporales claras)
    pasos = dias_diff if dias_diff > 0 else 1
    pred_arima = {
        'precipitacion': modelos['arima']['precipitacion'].forecast(steps=pasos).iloc[-1],
        'nivel_agua': modelos['arima']['nivel_agua'].forecast(steps=pasos).iloc[-1],
        'temperatura': modelos['arima']['temperatura'].forecast(steps=pasos).iloc[-1],
        'presion': modelos['arima']['presion'].forecast(steps=pasos).iloc[-1]
    }

    # Fusión de predicciones
    pred_fusion = {
        k: round(max(0, alpha * pred_arima[k] + (1 - alpha) * pred_rf[k]), 2)
        for k in ['precipitacion', 'nivel_agua', 'temperatura', 'presion']
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

    # Fechas futuras a partir del último dato
    ult = df.sort_values(by='fecha').iloc[-1]
    fecha_inicio = ult['fecha']
    fecha_fin = pd.to_datetime(f"{año}-{mes}-01")

    # 🧩 Validar si la fecha seleccionada es anterior al dataset
    if fecha_fin < fecha_inicio:
        interpretacion = "La fecha seleccionada es anterior a los datos disponibles. No se genera predicción futura."
        buf = io.BytesIO()
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, interpretacion, ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf), interpretacion

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

        # Actualizar valores de entrada si son usados
        if variable == 'precipitacion':
            lag_precipitacion = pred_ajustada
        elif variable == 'nivel_agua':
            lag_nivel_agua = pred_ajustada
        elif variable == 'temperatura':
            lag_temperatura = pred_ajustada
        elif variable == 'presion':
            lag_presion = pred_ajustada

    # Agregar proyección si existe
    if pred_fechas:
        plt.plot(pred_fechas, pred_valores, linestyle='-', color='gray', label="Proyección mensual futura")
        plt.scatter(pred_fechas[-1], pred_valores[-1], color='red', s=100, label=f"Predicción para {mes}/{año}")
        interpretacion = interpretar_variable(variable, pred_valores[-1])
    else:
        interpretacion = "No se pudo generar una predicción futura para la fecha seleccionada."

    plt.title(f"{variable.capitalize()} mensual y predicción")
    plt.xlabel("Fecha")
    plt.ylabel(variable.capitalize())
    plt.grid(True)
    plt.legend(loc='upper left')

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
def crear_grafica_lineal(df, variable='precipitacion_valor', color='blue'):
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    # Crear gráfico de líneas
    plt.figure(figsize=(12, 5))
    plt.plot(df['fecha'], df[variable], color=color, linewidth=2.0)
    plt.title(f"Histórico del valor - {variable.replace('_', ' ').capitalize()}", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel(variable.replace('_valor', '').capitalize(), fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)

    # Guardar en buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return Image.open(buf)

def crear_grafica_lineal_interactiva(df, variable='precipitacion_valor', color='blue'):
    df = df.sort_values(by='fecha')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['fecha'],
        y=df[variable],
        mode='lines+markers',
        name=variable.replace('_valor', '').capitalize(),
        line=dict(color=color),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f"📊 Evolución histórica de {variable.replace('_valor', '').capitalize()}",
        xaxis_title="Fecha",
        yaxis_title=variable.replace('_valor', '').capitalize(),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        width=900  # 👈 Añadimos ancho más grande
    )

    return fig

def crear_grafica_completa_interactiva(modelo_rf, df, variable, mes, año, color='blue', ventana_movil=30):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    df = df.sort_values(by='fecha')
    fecha_ultima = df['fecha'].max()
    fecha_objetivo = pd.to_datetime(f"{año}-{mes:02d}-01")

    # Línea real
    fechas = df['fecha']
    valores_reales = df[f"{variable}_valor"]

    # Promedio móvil
    promedio_movil = valores_reales.rolling(window=ventana_movil).mean()

    # Generar predicciones mes a mes hasta la fecha objetivo
    fechas_futuras = pd.date_range(start=fecha_ultima + pd.DateOffset(months=1), end=fecha_objetivo, freq='MS')
    pred_fechas, pred_valores = [], []

    ult = df.iloc[-1]
    lag_precipitacion = ult['precipitacion_valor']
    lag_temperatura = ult['temperatura_valor']
    lag_nivel_agua = ult['nivel_agua_valor']
    lag_presion = ult['presion_valor']

    for f in fechas_futuras:
        entrada = [[
            f.month,
            f.year,
            lag_precipitacion,
            lag_temperatura,
            lag_nivel_agua,
            lag_presion
        ]]
        pred = modelo_rf.predict(entrada)[0]
        pred = max(0, pred + np.random.normal(loc=0, scale=pred * 0.1))  # suavizar ruido
        pred_fechas.append(f)
        pred_valores.append(pred)

        # actualizar lags
        if variable == 'precipitacion':
            lag_precipitacion = pred
        elif variable == 'nivel_agua':
            lag_nivel_agua = pred
        elif variable == 'temperatura':
            lag_temperatura = pred
        elif variable == 'presion':
            lag_presion = pred

    # Gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fechas,
        y=valores_reales,
        mode="lines",
        name="Datos reales",
        line=dict(color=color)
    ))

    fig.add_trace(go.Scatter(
        x=fechas,
        y=promedio_movil,
        mode="lines",
        name=f"Promedio móvil {ventana_movil} días",
        line=dict(color='orange', dash='dash')
    ))

    if pred_fechas:
        fig.add_trace(go.Scatter(
            x=pred_fechas,
            y=pred_valores,
            mode="lines+markers",
            name="Predicción futura",
            line=dict(color='gray', dash='dot'),
            marker=dict(size=6, color='gray')
        ))

        # Punto rojo final
        fig.add_trace(go.Scatter(
            x=[pred_fechas[-1]],
            y=[pred_valores[-1]],
            mode="markers",
            name="Predicción seleccionada",
            marker=dict(size=12, color='red', symbol='circle')
        ))

    fig.update_layout(
        title=f"{variable.capitalize()} con tendencia y predicción",
        xaxis_title="Fecha",
        yaxis_title=variable.capitalize(),
        template="plotly_white",
        height=450,
        width=800
    )

    return fig


