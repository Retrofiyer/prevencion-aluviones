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
    temp = pred['temperatura']
    presion = pred['presion']
    
    # Calcular índice de riesgo combinado
    riesgo_precip = min(100, (precip / 70) * 100)  # 70mm fue el umbral crítico en 2022
    riesgo_nivel = min(100, (nivel / 35) * 100)    # 35cm umbral crítico estimado
    riesgo_combinado = (riesgo_precip + riesgo_nivel) / 2
    
    # USAR DIRECTAMENTE EL FALLBACK LOCAL MEJORADO
    return generar_interpretacion_combinada_local(pred, riesgo_combinado)
    
    # Código comentado temporalmente - se puede activar cuando Gemini funcione
    """
    prompt = (
        f"Eres un especialista en prevención de desastres naturales para La Gasca, Quito, Ecuador. "
        f"El 31 de enero de 2022 ocurrió un aluvión devastador con >70mm/h de lluvia que causó muertes y destrucción.\n\n"
        f"DATOS ACTUALES PREDICHOS:\n"
        f"🌧️ Precipitación: {precip} mm\n"
        f"🌊 Nivel de agua: {nivel} cm\n"
        f"🌡️ Temperatura: {temp} °C\n"
        f"📉 Presión: {presion} hPa\n"
        f"📊 Índice de riesgo combinado: {riesgo_combinado:.1f}%\n\n"
        f"PROPORCIONA UNA INTERPRETACIÓN INTEGRAL que incluya:\n\n"
        f"1. **EVALUACIÓN DE RIESGO**: ¿Qué tan cerca estamos de condiciones de diluvio?\n"
        f"2. **CONTEXTO HISTÓRICO**: Comparación con el evento del 31 de enero 2022\n"
        f"3. **FACTORES AGRAVANTES**: Cómo la temperatura y presión influyen en el riesgo\n"
        f"4. **ESCENARIOS POSIBLES**: Qué podría pasar si las condiciones empeoran\n"
        f"5. **ACCIONES INMEDIATAS**: Qué debe hacer la comunidad HOY\n"
        f"6. **VIGILANCIA COMUNITARIA**: Señales específicas que los vecinos deben observar\n"
        f"7. **PREPARACIÓN FAMILIAR**: Kit de emergencia y rutas de evacuación\n\n"
        f"Usa lenguaje claro, directo y ESPECÍFICO para La Gasca. Máximo 10-12 líneas."
    )
    
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return generar_interpretacion_combinada_local(pred, riesgo_combinado)
    """

def generar_interpretacion_combinada_local(pred, riesgo_combinado):
    """Interpretación detallada de respaldo cuando falla la API"""
    precip = pred['precipitacion']
    nivel = pred['nivel_agua']
    temp = pred['temperatura']
    presion = pred['presion']
    
    # Análisis contextual más detallado
    dias_hasta_critico = max(0, (70 - precip) / 10) if precip < 70 else 0
    factores_agravantes = []
    
    if temp > 20:
        factores_agravantes.append("temperatura alta intensifica evaporación y posterior condensación")
    if presion < 750:
        factores_agravantes.append("baja presión favorece formación de nubes de tormenta")
    if nivel > 15:
        factores_agravantes.append("quebradas ya con nivel elevado reducen margen de seguridad")
    
    factores_texto = " - " + ", ".join(factores_agravantes) if factores_agravantes else ""
    
    if riesgo_combinado < 15:
        return (
            f"✅ **RIESGO BAJO** ({riesgo_combinado:.1f}%): **SITUACIÓN CONTROLADA** - Precipitación {precip}mm y nivel {nivel}cm en rangos seguros. "
            f"Distancia al umbral crítico: ~{dias_hasta_critico:.1f} días de lluvia intensa{factores_texto}. "
            f"**APROVECHAR PARA**: Limpieza de canales, revisión de desagües, actualización de kit de emergencia, "
            f"coordinación vecinal. **VIGILANCIA**: Monitoreo diario de quebradas, pronósticos meteorológicos oficiales."
        )
    elif riesgo_combinado < 30:
        return (
            f"⚠️ **RIESGO MODERADO** ({riesgo_combinado:.1f}%): **VIGILANCIA ACTIVA** - Con {precip}mm lluvia y {nivel}cm nivel, "
            f"estamos a {70-precip:.1f}mm del umbral crítico (desastre 2022){factores_texto}. "
            f"**ACCIONES INMEDIATAS**: Verificar rutas de evacuación familiares, coordinar con vecinos sistemas de alerta, "
            f"mantener radio/celular cargado, documentos importantes en bolsa impermeable. **OBSERVAR**: Cambios súbitos en quebradas, "
            f"ruidos de arrastre de piedras, color del agua (transparente→marrón), crecimiento de caudal."
        )
    elif riesgo_combinado < 50:
        return (
            f"🚨 **RIESGO ALTO** ({riesgo_combinado:.1f}%): **ALERTA PREVENTIVA** - Precipitación {precip}mm se aproxima peligrosamente "
            f"al umbral del desastre del 31 enero 2022 (>70mm/h){factores_texto}. **RIESGO REAL**: Saturación acelerada del suelo, "
            f"crecimiento exponencial de caudales. **PREPARAR EVACUACIÓN**: Vehículo listo, combustible, ruta definida hacia terreno alto. "
            f"**COMUNICAR**: Situación a familiares, vecinos vulnerables, autoridades locales. **SEÑALES CRÍTICAS**: Rugido creciente en quebradas, "
            f"espuma en el agua, vibración en puentes, animales inquietos."
        )
    elif riesgo_combinado < 75:
        return (
            f"🔴 **RIESGO CRÍTICO** ({riesgo_combinado:.1f}%): **EMERGENCIA PREVENTIVA** - ¡Condiciones peligrosas similares al aluvión de 2022! "
            f"Precipitación {precip}mm y nivel {nivel}cm indican PELIGRO INMINENTE{factores_texto}. "
            f"**EVACUACIÓN PREVENTIVA RECOMENDADA** especialmente para: adultos mayores, niños, personas con discapacidad, viviendas cercanas a quebradas. "
            f"**ACCIONES CRÍTICAS**: Alertar 911, comunicar emergencia a vecinos, documentos y medicinas esenciales listos, "
            f"identificar refugios seguros (colegios, iglesias en terreno alto). **PELIGRO EXTREMO**: Rugido ensordecedor, temblor del suelo, olor intenso a tierra."
        )
    else:
        return (
            f"⚡ **EMERGENCIA EXTREMA** ({riesgo_combinado:.1f}%): **ALUVIÓN PROBABLE** - ¡EVACUACIÓN INMEDIATA! "
            f"Condiciones IGUALES O PEORES al desastre del 31 enero 2022. **PELIGRO MORTAL INMINENTE**{factores_texto}. "
            f"**ACTUAR AHORA**: Buscar terreno alto (>100m de quebradas), llamar 911 - EMERGENCIA MAYOR, "
            f"NO intentar rescatar pertenencias, alejarse inmediatamente de cauces y laderas inestables. "
            f"**SÍNTOMAS DE ALUVIÓN ACTIVO**: Rugido como tren, suelo temblando, rocas gigantes rodando, "
            f"árboles cayendo, corte súbito de servicios. **REFUGIO**: Estructuras sólidas en terreno alto, comunicación constante con emergencias."
        )

def interpretar_variable(variable, valor):
    # Definir umbrales críticos para La Gasca basados en el evento del 31 de enero de 2022
    umbrales = {
        'precipitacion': {
            'bajo': 10,
            'moderado': 30, 
            'alto': 50,
            'critico': 70,
            'extremo': 100
        },
        'nivel_agua': {
            'bajo': 5,
            'moderado': 15,
            'alto': 25,
            'critico': 35,
            'extremo': 50
        },
        'temperatura': {
            'bajo': 12,
            'moderado': 18,
            'alto': 24,
            'critico': 30,
            'extremo': 35
        },
        'presion': {
            'bajo': 740,
            'moderado': 760,
            'alto': 780,
            'critico': 800,
            'extremo': 820
        }
    }
    
    def obtener_nivel_riesgo(variable, valor, umbrales):
        if valor <= umbrales[variable]['bajo']:
            return 'bajo'
        elif valor <= umbrales[variable]['moderado']:
            return 'moderado'
        elif valor <= umbrales[variable]['alto']:
            return 'alto'
        elif valor <= umbrales[variable]['critico']:
            return 'crítico'
        else:
            return 'extremo'
    
    nivel = obtener_nivel_riesgo(variable, valor, umbrales)
    
    # USAR DIRECTAMENTE EL FALLBACK LOCAL MEJORADO
    return generar_interpretacion_local(variable, valor, nivel)
    
    # Código comentado temporalmente - se puede activar cuando Gemini funcione
    """
    prompt = (
        f"Como experto en prevención de desastres naturales especializado en La Gasca (Quito), "
        f"proporciona una interpretación COMPLETA y EDUCATIVA sobre:\n\n"
        f"Variable: {variable}\n"
        f"Valor actual: {valor}\n"
        f"Nivel de riesgo detectado: {nivel}\n\n"
        f"Tu respuesta debe incluir:\n"
        f"1. **Contexto del valor**: Qué significa este número en términos simples (usa analogías cotidianas)\n"
        f"2. **Proximidad al peligro**: ¿Qué tan cerca estamos de condiciones de diluvio? "
        f"(El 31 de enero de 2022 hubo un aluvión devastador con >70mm/h de lluvia)\n"
        f"3. **Factores que influyen**: Qué otros elementos podrían empeorar la situación "
        f"(saturación del suelo, obstrucción de quebradas, pendiente, etc.)\n"
        f"4. **Ejemplos prácticos**: Comparaciones con situaciones conocidas o eventos históricos\n"
        f"5. **Acciones específicas**: Qué debe hacer la comunidad AHORA según este nivel\n"
        f"6. **Señales de alerta**: Qué observar para detectar empeoramiento\n\n"
        f"Máximo 6-8 líneas, lenguaje claro y directo para la comunidad."
    )
    
    try:
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Fallback con interpretación local más detallada
        return generar_interpretacion_local(variable, valor, nivel)
    """

def generar_interpretacion_local(variable, valor, nivel):
    """Función de respaldo para generar interpretaciones detalladas sin API"""
    interpretaciones = {
        'precipitacion': {
            'bajo': f"💧 **LLUVIA LIGERA** ({valor} mm): Como rocío matutino o llovizna suave. Estamos a {70-valor} mm del umbral crítico de diluvio (>70 mm/h causó el desastre de 2022). **QUÉ HACER**: Momento ideal para limpiar canales y desagües. Revisar que nada obstruya quebradas. Mantener kit de emergencia actualizado. **VIGILAR**: Acumulación en 24h, saturación del suelo en laderas, cambios de color en quebradas.",
            'moderado': f"🌧️ **LLUVIA MODERADA** ({valor} mm): Como ducha normal, suelo comenzando a saturarse. Solo {70-valor} mm nos separan del nivel de ALERTA ROJA. **PELIGRO CRECIENTE**: Si continúa por horas puede saturar completamente el suelo en pendientes. **ACCIONES**: Coordinar con vecinos, verificar rutas de evacuación, tener radio/celular cargado. **OBSERVAR**: Ruido de arrastre en quebradas, agua turbia, crecimiento del caudal.",
            'alto': f"⚠️ **LLUVIA INTENSA** ({valor} mm): Como manguera abierta, PELIGRO REAL. Solo {70-valor} mm del umbral del desastre de 2022. Suelo saturándose rápidamente, quebradas subiendo. **RIESGO INMINENTE**: Deslizamientos en laderas, desbordamiento de cauces. **EVACUAR PREVENTIVAMENTE** de zonas bajas. Alejarse de quebradas. Documentos listos. **ALERTA MÁXIMA**: Ruidos extraños, agua lodosa, piedras rodando.",
            'critico': f"🚨 **ALERTA ROJA** ({valor} mm): NIVEL CRÍTICO alcanzado. Condiciones similares al aluvión del 31 enero 2022 que causó muertes. Riesgo INMINENTE de desbordamiento masivo. **EVACUACIÓN PREVENTIVA OBLIGATORIA**. Buscar terreno alto inmediatamente. Alejarse de quebradas y laderas. **EMERGENCIA**: Llamar 911, alertar a vecinos. **SEÑALES DE PELIGRO EXTREMO**: Rugido de agua, piedras grandes rodando, grietas en el suelo.",
            'extremo': f"🔴 **EVACUACIÓN INMEDIATA** ({valor} mm): SUPERA el umbral del desastre 2022. Condiciones EXTREMAS de diluvio activo. ALUVIÓN EN CURSO probable. **ACTUAR YA**: Terreno alto, lejos de cauces. 911 - EMERGENCIA. **PELIGRO MORTAL**: No permanecer en viviendas cerca de quebradas. **SÍNTOMAS DE ALUVIÓN**: Rugido ensordecedor, temblor del suelo, olor a tierra mojada intenso, animales huyendo."
        },
        'nivel_agua': {
            'bajo': f"🌊 **NIVEL NORMAL** ({valor} cm): Quebrada en capacidad adecuada. Drenaje funcionando bien. **OPORTUNIDAD**: Momento perfecto para limpieza de canales, remoción de escombros, revisión de infraestructura. **VIGILANCIA COMUNITARIA**: Establecer turnos de observación, identificar puntos críticos. **PREPARACIÓN**: Actualizar rutas de evacuación, revisar kit de emergencia familiar.",
            'moderado': f"📈 **NIVEL CRECIENTE** ({valor} cm): Quebradas llenándose gradualmente. Aún manejable pero requiere atención. **FACTORES DE RIESGO**: Lluvia sostenida puede saturar capacidad en 2-4 horas. **VIGILAR**: Ruido de piedras arrastrándose, cambio de color del agua (transparente→marrón), aumento de velocidad. **ACCIONES**: Alejar vehículos de cauces, verificar que niños no jueguen cerca del agua.",
            'alto': f"⚠️ **NIVEL PREOCUPANTE** ({valor} cm): Quebrada cerca de capacidad máxima. Riesgo de desbordamiento en 1-2 horas si continúa subiendo. **PELIGRO REAL**: Erosión de orillas, arrastre de objetos grandes. **PREPARACIÓN INMEDIATA**: Evacuar preventivamente zonas bajas, tener vehículos listos para salir, documentos importantes en bolsa impermeable. **OBSERVAR**: Espuma en el agua, ruido creciente, vibración en puentes.",
            'critico': f"🚨 **NIVEL CRÍTICO** ({valor} cm): Capacidad de quebrada AL LÍMITE. Desbordamiento INMINENTE en minutos u horas. **EVACUACIÓN PREVENTIVA OBLIGATORIA** de zonas bajas. **PELIGRO EXTREMO**: Arrastre de rocas grandes, socavación de cimientos, colapso de puentes. **ACTUAR INMEDIATAMENTE**: Terreno alto, alejarse 100+ metros de cauces, llamar emergencias 911.",
            'extremo': f"🔴 **DESBORDAMIENTO ACTIVO** ({valor} cm): NIVEL EXTREMO - Aluvión en desarrollo. Quebrada desbordando o a punto de hacerlo. **EVACUACIÓN INMEDIATA**: Buscar terreno alto YA. **EMERGENCIA 911**: Reportar situación crítica. **PELIGRO MORTAL**: Flujo de escombros, arrastre de vehículos, destrucción de infraestructura. **NO INTENTAR cruzar cauces o rescatar objetos**."
        },
        'temperatura': {
            'bajo': f"🌡️ **TEMPERATURA BAJA** ({valor}°C): Condiciones frías que pueden intensificar efectos de lluvia. **CONTEXTO CLIMÁTICO**: Aire frío retiene menos humedad, puede generar lluvias más prolongadas. **CONSIDERACIONES**: Mayor riesgo de hipotermia en emergencias, suelo más compacto (menos absorción). **PREPARACIÓN**: Ropa abrigada en kit de emergencia, mantas térmicas, combustible para calefacción.",
            'moderado': f"🌡️ **TEMPERATURA NORMAL** ({valor}°C): Condiciones típicas de Quito. **VENTAJA**: Temperatura estable facilita evacuaciones y rescates. Suelo con capacidad normal de absorción. **MANTENER**: Vigilancia normal, preparación estándar de emergencias. **RECORDAR**: Cambios bruscos de temperatura pueden indicar frentes meteorológicos intensos.",
            'alto': f"🌡️ **TEMPERATURA ELEVADA** ({valor}°C): Calor inusual para La Gasca puede indicar sistemas meteorológicos intensos. **ALERTA**: Aire caliente retiene más humedad, posibles tormentas más fuertes. Suelo seco absorbe menos agua inicialmente. **PREPARACIÓN**: Hidratación en kit de emergencia, protección solar, considerar mayor volatilidad climática.",
            'critico': f"🌡️ **TEMPERATURA MUY ALTA** ({valor}°C): Condiciones excepcionales que pueden preceder eventos climáticos extremos. **CONTEXTO**: Gradientes térmicos fuertes generan inestabilidad atmosférica severa. **ALERTA MÁXIMA**: Posibles tormentas supercélulas, granizo, vientos fuertes. **PREPARACIÓN ESPECIAL**: Refugio sólido, comunicaciones de emergencia.",
            'extremo': f"🌡️ **TEMPERATURA EXTREMA** ({valor}°C): Condiciones anómalas que requieren máxima precaución. **PELIGRO**: Sistemas meteorológicos severos probable. **EMERGENCIA CLIMÁTICA**: Mantenerse informado via radio oficial, refugio seguro, evitar exposición prolongada al exterior."
        },
        'presion': {
            'bajo': f"📉 **PRESIÓN BAJA** ({valor} hPa): Sistema de baja presión puede indicar aproximación de frente lluvioso. **CONTEXTO METEOROLÓGICO**: Aire ascendente, condensación, nubes cumulonimbus. **VIGILANCIA**: Posible intensificación de lluvias en 6-12 horas. **PREPARACIÓN**: Revisar pronósticos oficiales, tener plan de contingencia listo.",
            'moderado': f"📉 **PRESIÓN NORMAL** ({valor} hPa): Condiciones atmosféricas estables para Quito (2800 msnm). **VENTAJA**: Menos probabilidad de cambios meteorológicos súbitos. **MANTENER**: Vigilancia estándar, preparación normal de emergencias. **APROVECHAR**: Momento óptimo para mantenimiento preventivo.",
            'alto': f"📉 **PRESIÓN ALTA** ({valor} hPa): Alta presión puede indicar estabilidad meteorológica temporal. **CONTEXTO**: Aire descendente, despeje de nubes. Sin embargo, cambios bruscos pueden generar tormentas posteriores. **OPORTUNIDAD**: Realizar trabajos de prevención, limpieza de canales.",
            'critico': f"📉 **PRESIÓN MUY ALTA** ({valor} hPa): Condiciones atmosféricas inusuales. **ALERTA**: Gradientes de presión fuertes pueden preceder cambios meteorológicos súbitos y severos. **VIGILANCIA ESPECIAL**: Monitorear pronósticos oficiales cada hora, tener comunicaciones listas.",
            'extremo': f"📉 **PRESIÓN EXTREMA** ({valor} hPa): Condiciones atmosféricas anómalas. **EMERGENCIA METEOROLÓGICA**: Posibles fenómenos severos (tornados, granizo, vientos destructivos). **REFUGIO INMEDIATO**: Estructura sólida, comunicación con autoridades, evitar salir al exterior."
        }
    }
    
    if variable in interpretaciones and nivel in interpretaciones[variable]:
        return interpretaciones[variable][nivel]
    else:
        return f"⚠️ **VALOR ANÓMALO** {variable}: {valor} - Nivel {nivel}. Condiciones fuera de parámetros normales. **PRECAUCIÓN MÁXIMA**: Consultar con autoridades meteorológicas. Mantener vigilancia comunitaria extrema y preparación para evacuación."
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
        height=400,
        width=900  # 👈 Ampliamos significativamente el ancho
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
        height=400,
        width=950  # 👈 Gráficas más anchas
    )

    return fig


