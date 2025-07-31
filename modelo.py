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
# ENTRENAMIENTO DEL MODELO MEJORADO
# ===============================
def entrenar_modelo():
    """Entrenar modelo tradicional (mantenido para compatibilidad)"""
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

def entrenar_modelo_mejorado():
    """Entrena múltiples modelos para crear un ensemble más robusto"""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    
    df = pd.read_csv(CSV_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Features expandidas
    df['Mes'] = df['fecha'].dt.month
    df['Año'] = df['fecha'].dt.year
    df['Día_año'] = df['fecha'].dt.dayofyear
    df['Estacion'] = df['fecha'].dt.month.map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
                                               6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
    
    # Lags múltiples
    for lag in [1, 2, 3, 7]:  # 1, 2, 3 días y 1 semana
        df[f'precipitacion_lag{lag}'] = df['precipitacion_valor'].shift(lag)
        df[f'temperatura_lag{lag}'] = df['temperatura_valor'].shift(lag)
        df[f'nivel_agua_lag{lag}'] = df['nivel_agua_valor'].shift(lag)
        df[f'presion_lag{lag}'] = df['presion_valor'].shift(lag)
    
    # Medias móviles
    for window in [7, 15, 30]:
        df[f'precip_ma{window}'] = df['precipitacion_valor'].rolling(window).mean()
        df[f'temp_ma{window}'] = df['temperatura_valor'].rolling(window).mean()
    
    df = df.dropna()
    
    # Features seleccionadas
    lag_features = [col for col in df.columns if 'lag' in col or 'ma' in col]
    features = ['Mes', 'Año', 'Día_año', 'Estacion'] + lag_features
    
    # Ensemble de modelos
    modelos_ensemble = {}
    for variable in ['precipitacion', 'nivel_agua', 'temperatura', 'presion']:
        col_target = f'{variable}_valor'
        
        modelos_ensemble[variable] = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gbr': GradientBoostingRegressor(n_estimators=150, max_depth=6, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Entrenar cada modelo
        X = df[features]
        y = df[col_target]
        
        for modelo_name, modelo in modelos_ensemble[variable].items():
            modelo.fit(X, y)
    
    return modelos_ensemble, df, features


# ===============================
# FUNCIÓN DE PRECISIÓN TEMPORAL MEJORADA
# ===============================
def calcular_precision_temporal(fecha_objetivo, fecha_ultima):
    """Función original mantenida para compatibilidad"""
    dias_diff = (fecha_objetivo - fecha_ultima).days
    if dias_diff <= 0:
        return 1.0
    max_dias = 365 * 2
    precision = max(0.3, 1 - (dias_diff / max_dias))
    return round(precision * 100, 1)

def calcular_precision_temporal_mejorada(fecha_objetivo, fecha_ultima, variables_pred, datos_historicos):
    """
    Calcula precisión considerando múltiples factores para mejorar confiabilidad
    """
    dias_diff = (fecha_objetivo - fecha_ultima).days
    if dias_diff <= 0:
        return 100.0
    
    # 1. Decaimiento base más suave
    max_dias = 365 * 3  # Extender horizonte temporal
    precision_base = max(0.4, 1 - (dias_diff / max_dias) ** 0.8)  # Función de decaimiento suavizada
    
    # 2. Ajuste por estacionalidad (patrones recurrentes)
    mes_objetivo = fecha_objetivo.month
    precision_estacional = calcular_precision_estacional(mes_objetivo, datos_historicos)
    
    # 3. Ajuste por variabilidad histórica de las variables
    factor_variabilidad = calcular_factor_variabilidad(variables_pred, datos_historicos)
    
    # 4. Ajuste por tendencias históricas
    factor_tendencia = calcular_factor_tendencia(fecha_objetivo, datos_historicos)
    
    # 5. Combinación ponderada
    precision_final = (
        precision_base * 0.4 +
        precision_estacional * 0.3 +
        factor_variabilidad * 0.2 +
        factor_tendencia * 0.1
    )
    
    return round(min(100, max(30, precision_final * 100)), 1)

def calcular_precision_estacional(mes, datos_historicos):
    """Evalúa qué tan predecible es el clima en ese mes"""
    try:
        datos_mes = datos_historicos[datos_historicos['fecha'].dt.month == mes]
        
        if len(datos_mes) < 10:  # Pocos datos
            return 0.5
        
        # Calcular variabilidad estacional
        cv_precip = datos_mes['precipitacion_valor'].std() / (datos_mes['precipitacion_valor'].mean() + 0.1)
        cv_temp = datos_mes['temperatura_valor'].std() / (datos_mes['temperatura_valor'].mean() + 0.1)
        
        # Menor variabilidad = mayor precisión
        precision_est = 1 / (1 + (cv_precip + cv_temp) / 2)
        return min(1.0, precision_est)
    except:
        return 0.5

def calcular_factor_variabilidad(variables_pred, datos_historicos):
    """Evalúa si las variables predichas están en rangos históricos normales"""
    try:
        factor = 1.0
        
        for var, valor in variables_pred.items():
            if var == 'precipitacion':
                col = 'precipitacion_valor'
                # Para precipitación, valores extremos reducen confianza
                percentil_95 = datos_historicos[col].quantile(0.95)
                if valor > percentil_95:
                    factor *= 0.7  # Reducir confianza para valores extremos
            
            elif var == 'temperatura':
                col = 'temperatura_valor'
                mean_temp = datos_historicos[col].mean()
                std_temp = datos_historicos[col].std()
                # Valores muy alejados de la media reducen confianza
                z_score = abs(valor - mean_temp) / std_temp
                if z_score > 2:
                    factor *= 0.8
        
        return factor
    except:
        return 1.0

def calcular_factor_tendencia(fecha_objetivo, datos_historicos):
    """Evalúa tendencias de largo plazo para mejorar predicciones lejanas"""
    try:
        # Calcular tendencias por año
        datos_anuales = datos_historicos.groupby(datos_historicos['fecha'].dt.year).agg({
            'precipitacion_valor': 'mean',
            'temperatura_valor': 'mean'
        })
        
        if len(datos_anuales) < 3:
            return 0.5
        
        # Calcular tendencia lineal
        años = datos_anuales.index.values
        precip_values = datos_anuales['precipitacion_valor'].values
        
        # Si hay tendencia clara, la predicción a largo plazo es más confiable
        correlation = abs(np.corrcoef(años, precip_values)[0, 1])
        
        # Mayor correlación temporal = mayor confianza en predicciones lejanas
        return min(1.0, 0.5 + correlation * 0.5)
    except:
        return 0.5

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

def predecir_variables_ensemble(modelos_ensemble, mes, año, dia, ultimos_valores, df, features):
    """Predicción mejorada usando ensemble de modelos"""
    fecha_pred = pd.Timestamp(f"{año}-{mes}-{dia}")
    fecha_ultima = df['fecha'].max()
    dias_diff = (fecha_pred - fecha_ultima).days
    
    # Preparar entrada con todas las features
    entrada_dict = {
        'Mes': mes,
        'Año': año,
        'Día_año': fecha_pred.dayofyear,
        'Estacion': {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}[mes]
    }
    
    # Agregar lags (usar últimos valores disponibles)
    for lag in [1, 2, 3, 7]:
        for var in ['precipitacion', 'temperatura', 'nivel_agua', 'presion']:
            entrada_dict[f'{var}_lag{lag}'] = ultimos_valores[f'{var}_valor']
    
    # Agregar medias móviles (calcular desde datos históricos)
    for window in [7, 15, 30]:
        entrada_dict[f'precip_ma{window}'] = df['precipitacion_valor'].tail(window).mean()
        entrada_dict[f'temp_ma{window}'] = df['temperatura_valor'].tail(window).mean()
    
    # Crear DataFrame de entrada
    entrada_df = pd.DataFrame([entrada_dict])
    entrada_df = entrada_df.reindex(columns=features, fill_value=0)
    
    # Predicciones ensemble
    predicciones_ensemble = {}
    intervalos_confianza = {}
    
    for variable in ['precipitacion', 'nivel_agua', 'temperatura', 'presion']:
        predicciones = []
        
        # Obtener predicción de cada modelo
        for modelo_name, modelo in modelos_ensemble[variable].items():
            pred = modelo.predict(entrada_df)[0]
            predicciones.append(pred)
        
        # Promedio ponderado (RF tiene más peso por ser más robusto)
        pesos = {'rf': 0.5, 'gbr': 0.3, 'ridge': 0.2}
        pred_final = sum(pred * pesos[modelo_name] 
                        for pred, modelo_name in zip(predicciones, pesos.keys()))
        
        # Calcular intervalo de confianza basado en dispersión de modelos
        std_predicciones = np.std(predicciones)
        intervalo_inf = pred_final - 1.96 * std_predicciones
        intervalo_sup = pred_final + 1.96 * std_predicciones
        
        predicciones_ensemble[variable] = max(0, pred_final)
        intervalos_confianza[variable] = (max(0, intervalo_inf), intervalo_sup)
    
    # Calcular precisión mejorada
    precision = calcular_precision_temporal_mejorada(
        fecha_pred, fecha_ultima, predicciones_ensemble, df
    )
    
    # Generar série temporal de precisión
    fechas_precision = pd.date_range(start=fecha_ultima + pd.Timedelta(days=1), 
                                    end=fecha_pred, freq='D')
    valores_precision = [calcular_precision_temporal_mejorada(f, fecha_ultima, predicciones_ensemble, df) 
                        for f in fechas_precision]
    
    return predicciones_ensemble, intervalos_confianza, precision, fechas_precision, valores_precision

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
            f"✅ <b>RIESGO BAJO</b> ({riesgo_combinado:.1f}%): <b>SITUACIÓN CONTROLADA</b> - Precipitación {precip}mm y nivel {nivel}cm en rangos seguros. "
            f"Distancia al umbral crítico: ~{dias_hasta_critico:.1f} días de lluvia intensa{factores_texto}. "
            f"<b>APROVECHAR PARA</b>: Limpieza de canales, revisión de desagües, actualización de kit de emergencia, "
            f"coordinación vecinal. <b>VIGILANCIA</b>: Monitoreo diario de quebradas, pronósticos meteorológicos oficiales."
        )
    elif riesgo_combinado < 30:
        return (
            f"⚠️ <b>RIESGO MODERADO</b> ({riesgo_combinado:.1f}%): <b>VIGILANCIA ACTIVA</b> - Con {precip}mm lluvia y {nivel}cm nivel, "
            f"estamos a {70-precip:.1f}mm del umbral crítico (desastre 2022){factores_texto}. "
            f"<b>ACCIONES INMEDIATAS</b>: Verificar rutas de evacuación familiares, coordinar con vecinos sistemas de alerta, "
            f"mantener radio/celular cargado, documentos importantes en bolsa impermeable. <b>OBSERVAR</b>: Cambios súbitos en quebradas, "
            f"ruidos de arrastre de piedras, color del agua (transparente→marrón), crecimiento de caudal."
        )
    elif riesgo_combinado < 50:
        return (
            f"🚨 <b>RIESGO ALTO</b> ({riesgo_combinado:.1f}%): <b>ALERTA PREVENTIVA</b> - Precipitación {precip}mm se aproxima peligrosamente "
            f"al umbral del desastre del 31 enero 2022 (>70mm/h){factores_texto}. <b>RIESGO REAL</b>: Saturación acelerada del suelo, "
            f"crecimiento exponencial de caudales. <b>PREPARAR EVACUACIÓN</b>: Vehículo listo, combustible, ruta definida hacia terreno alto. "
            f"<b>COMUNICAR</b>: Situación a familiares, vecinos vulnerables, autoridades locales. <b>SEÑALES CRÍTICAS</b>: Rugido creciente en quebradas, "
            f"espuma en el agua, vibración en puentes, animales inquietos."
        )
    elif riesgo_combinado < 75:
        return (
            f"🔴 <b>RIESGO CRÍTICO</b> ({riesgo_combinado:.1f}%): <b>EMERGENCIA PREVENTIVA</b> - ¡Condiciones peligrosas similares al aluvión de 2022! "
            f"Precipitación {precip}mm y nivel {nivel}cm indican PELIGRO INMINENTE{factores_texto}. "
            f"<b>EVACUACIÓN PREVENTIVA RECOMENDADA</b> especialmente para: adultos mayores, niños, personas con discapacidad, viviendas cercanas a quebradas. "
            f"<b>ACCIONES CRÍTICAS</b>: Alertar 911, comunicar emergencia a vecinos, documentos y medicinas esenciales listos, "
            f"identificar refugios seguros (colegios, iglesias en terreno alto). <b>PELIGRO EXTREMO</b>: Rugido ensordecedor, temblor del suelo, olor intenso a tierra."
        )
    else:
        return (
            f"⚡ <b>EMERGENCIA EXTREMA</b> ({riesgo_combinado:.1f}%): <b>ALUVIÓN PROBABLE</b> - ¡EVACUACIÓN INMEDIATA! "
            f"Condiciones IGUALES O PEORES al desastre del 31 enero 2022. <b>PELIGRO MORTAL INMINENTE</b>{factores_texto}. "
            f"<b>ACTUAR AHORA</b>: Buscar terreno alto (>100m de quebradas), llamar 911 - EMERGENCIA MAYOR, "
            f"NO intentar rescatar pertenencias, alejarse inmediatamente de cauces y laderas inestables. "
            f"<b>SÍNTOMAS DE ALUVIÓN ACTIVO</b>: Rugido como tren, suelo temblando, rocas gigantes rodando, "
            f"árboles cayendo, corte súbito de servicios. <b>REFUGIO</b>: Estructuras sólidas en terreno alto, comunicación constante con emergencias."
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
            'bajo': f"💧 <b>LLUVIA LIGERA</b> ({valor} mm): Como rocío matutino o llovizna suave. Estamos a {70-valor} mm del umbral crítico de diluvio (>70 mm/h causó el desastre de 2022). <b>QUÉ HACER</b>: Momento ideal para limpiar canales y desagües. Revisar que nada obstruya quebradas. Mantener kit de emergencia actualizado. <b>VIGILAR</b>: Acumulación en 24h, saturación del suelo en laderas, cambios de color en quebradas.",
            'moderado': f"🌧️ <b>LLUVIA MODERADA</b> ({valor} mm): Como ducha normal, suelo comenzando a saturarse. Solo {70-valor} mm nos separan del nivel de ALERTA ROJA. <b>PELIGRO CRECIENTE</b>: Si continúa por horas puede saturar completamente el suelo en pendientes. <b>ACCIONES</b>: Coordinar con vecinos, verificar rutas de evacuación, tener radio/celular cargado. <b>OBSERVAR</b>: Ruido de arrastre en quebradas, agua turbia, crecimiento del caudal.",
            'alto': f"⚠️ <b>LLUVIA INTENSA</b> ({valor} mm): Como manguera abierta, PELIGRO REAL. Solo {70-valor} mm del umbral del desastre de 2022. Suelo saturándose rápidamente, quebradas subiendo. <b>RIESGO INMINENTE</b>: Deslizamientos en laderas, desbordamiento de cauces. <b>EVACUAR PREVENTIVAMENTE</b> de zonas bajas. Alejarse de quebradas. Documentos listos. <b>ALERTA MÁXIMA</b>: Ruidos extraños, agua lodosa, piedras rodando.",
            'critico': f"🚨 <b>ALERTA ROJA</b> ({valor} mm): NIVEL CRÍTICO alcanzado. Condiciones similares al aluvión del 31 enero 2022 que causó muertes. Riesgo INMINENTE de desbordamiento masivo. <b>EVACUACIÓN PREVENTIVA OBLIGATORIA</b>. Buscar terreno alto inmediatamente. Alejarse de quebradas y laderas. <b>EMERGENCIA</b>: Llamar 911, alertar a vecinos. <b>SEÑALES DE PELIGRO EXTREMO</b>: Rugido de agua, piedras grandes rodando, grietas en el suelo.",
            'extremo': f"🔴 <b>EVACUACIÓN INMEDIATA</b> ({valor} mm): SUPERA el umbral del desastre 2022. Condiciones EXTREMAS de diluvio activo. ALUVIÓN EN CURSO probable. <b>ACTUAR YA</b>: Terreno alto, lejos de cauces. 911 - EMERGENCIA. <b>PELIGRO MORTAL</b>: No permanecer en viviendas cerca de quebradas. <b>SÍNTOMAS DE ALUVIÓN</b>: Rugido ensordecedor, temblor del suelo, olor a tierra mojada intenso, animales huyendo."
        },
        'nivel_agua': {
            'bajo': f"🌊 <b>NIVEL NORMAL</b> ({valor} cm): Quebrada en capacidad adecuada. Drenaje funcionando bien. <b>OPORTUNIDAD</b>: Momento perfecto para limpieza de canales, remoción de escombros, revisión de infraestructura. <b>VIGILANCIA COMUNITARIA</b>: Establecer turnos de observación, identificar puntos críticos. <b>PREPARACIÓN</b>: Actualizar rutas de evacuación, revisar kit de emergencia familiar.",
            'moderado': f"📈 <b>NIVEL CRECIENTE</b> ({valor} cm): Quebradas llenándose gradualmente. Aún manejable pero requiere atención. <b>FACTORES DE RIESGO</b>: Lluvia sostenida puede saturar capacidad en 2-4 horas. <b>VIGILAR</b>: Ruido de piedras arrastrándose, cambio de color del agua (transparente→marrón), aumento de velocidad. <b>ACCIONES</b>: Alejar vehículos de cauces, verificar que niños no jueguen cerca del agua.",
            'alto': f"⚠️ <b>NIVEL PREOCUPANTE</b> ({valor} cm): Quebrada cerca de capacidad máxima. Riesgo de desbordamiento en 1-2 horas si continúa subiendo. <b>PELIGRO REAL</b>: Erosión de orillas, arrastre de objetos grandes. <b>PREPARACIÓN INMEDIATA</b>: Evacuar preventivamente zonas bajas, tener vehículos listos para salir, documentos importantes en bolsa impermeable. <b>OBSERVAR</b>: Espuma en el agua, ruido creciente, vibración en puentes.",
            'critico': f"🚨 <b>NIVEL CRÍTICO</b> ({valor} cm): Capacidad de quebrada AL LÍMITE. Desbordamiento INMINENTE en minutos u horas. <b>EVACUACIÓN PREVENTIVA OBLIGATORIA</b> de zonas bajas. <b>PELIGRO EXTREMO</b>: Arrastre de rocas grandes, socavación de cimientos, colapso de puentes. <b>ACTUAR INMEDIATAMENTE</b>: Terreno alto, alejarse 100+ metros de cauces, llamar emergencias 911.",
            'extremo': f"🔴 <b>DESBORDAMIENTO ACTIVO</b> ({valor} cm): NIVEL EXTREMO - Aluvión en desarrollo. Quebrada desbordando o a punto de hacerlo. <b>EVACUACIÓN INMEDIATA</b>: Buscar terreno alto YA. <b>EMERGENCIA 911</b>: Reportar situación crítica. <b>PELIGRO MORTAL</b>: Flujo de escombros, arrastre de vehículos, destrucción de infraestructura. <b>NO INTENTAR cruzar cauces o rescatar objetos</b>."
        },
        'temperatura': {
            'bajo': f"🌡️ <b>TEMPERATURA BAJA</b> ({valor}°C): Condiciones frías que pueden intensificar efectos de lluvia. <b>CONTEXTO CLIMÁTICO</b>: Aire frío retiene menos humedad, puede generar lluvias más prolongadas. <b>CONSIDERACIONES</b>: Mayor riesgo de hipotermia en emergencias, suelo más compacto (menos absorción). <b>PREPARACIÓN</b>: Ropa abrigada en kit de emergencia, mantas térmicas, combustible para calefacción.",
            'moderado': f"🌡️ <b>TEMPERATURA NORMAL</b> ({valor}°C): Condiciones típicas de Quito. <b>VENTAJA</b>: Temperatura estable facilita evacuaciones y rescates. Suelo con capacidad normal de absorción. <b>MANTENER</b>: Vigilancia normal, preparación estándar de emergencias. <b>RECORDAR</b>: Cambios bruscos de temperatura pueden indicar frentes meteorológicos intensos.",
            'alto': f"🌡️ <b>TEMPERATURA ELEVADA</b> ({valor}°C): Calor inusual para La Gasca puede indicar sistemas meteorológicos intensos. <b>ALERTA</b>: Aire caliente retiene más humedad, posibles tormentas más fuertes. Suelo seco absorbe menos agua inicialmente. <b>PREPARACIÓN</b>: Hidratación en kit de emergencia, protección solar, considerar mayor volatilidad climática.",
            'critico': f"🌡️ <b>TEMPERATURA MUY ALTA</b> ({valor}°C): Condiciones excepcionales que pueden preceder eventos climáticos extremos. <b>CONTEXTO</b>: Gradientes térmicos fuertes generan inestabilidad atmosférica severa. <b>ALERTA MÁXIMA</b>: Posibles tormentas supercélulas, granizo, vientos fuertes. <b>PREPARACIÓN ESPECIAL</b>: Refugio sólido, comunicaciones de emergencia.",
            'extremo': f"🌡️ <b>TEMPERATURA EXTREMA</b> ({valor}°C): Condiciones anómalas que requieren máxima precaución. <b>PELIGRO</b>: Sistemas meteorológicos severos probable. <b>EMERGENCIA CLIMÁTICA</b>: Mantenerse informado via radio oficial, refugio seguro, evitar exposición prolongada al exterior."
        },
        'presion': {
            'bajo': f"📉 <b>PRESIÓN BAJA</b> ({valor} hPa): Sistema de baja presión puede indicar aproximación de frente lluvioso. <b>CONTEXTO METEOROLÓGICO</b>: Aire ascendente, condensación, nubes cumulonimbus. <b>VIGILANCIA</b>: Posible intensificación de lluvias en 6-12 horas. <b>PREPARACIÓN</b>: Revisar pronósticos oficiales, tener plan de contingencia listo.",
            'moderado': f"📉 <b>PRESIÓN NORMAL</b> ({valor} hPa): Condiciones atmosféricas estables para Quito (2800 msnm). <b>VENTAJA</b>: Menos probabilidad de cambios meteorológicos súbitos. <b>MANTENER</b>: Vigilancia estándar, preparación normal de emergencias. <b>APROVECHAR</b>: Momento óptimo para mantenimiento preventivo.",
            'alto': f"📉 <b>PRESIÓN ALTA</b> ({valor} hPa): Alta presión puede indicar estabilidad meteorológica temporal. <b>CONTEXTO</b>: Aire descendente, despeje de nubes. Sin embargo, cambios bruscos pueden generar tormentas posteriores. <b>OPORTUNIDAD</b>: Realizar trabajos de prevención, limpieza de canales.",
            'critico': f"📉 <b>PRESIÓN MUY ALTA</b> ({valor} hPa): Condiciones atmosféricas inusuales. <b>ALERTA</b>: Gradientes de presión fuertes pueden preceder cambios meteorológicos súbitos y severos. <b>VIGILANCIA ESPECIAL</b>: Monitorear pronósticos oficiales cada hora, tener comunicaciones listas.",
            'extremo': f"📉 <b>PRESIÓN EXTREMA</b> ({valor} hPa): Condiciones atmosféricas anómalas. <b>EMERGENCIA METEOROLÓGICA</b>: Posibles fenómenos severos (tornados, granizo, vientos destructivos). <b>REFUGIO INMEDIATO</b>: Estructura sólida, comunicación con autoridades, evitar salir al exterior."
        }
    }
    
    if variable in interpretaciones and nivel in interpretaciones[variable]:
        return interpretaciones[variable][nivel]
    else:
        return f"⚠️ <b>VALOR ANÓMALO</b> {variable}: {valor} - Nivel {nivel}. Condiciones fuera de parámetros normales. <b>PRECAUCIÓN MÁXIMA</b>: Consultar con autoridades meteorológicas. Mantener vigilancia comunitaria extrema y preparación para evacuación."
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

# ===============================
# INTERPRETACIÓN MEJORADA DE PRECISIÓN
# ===============================
def interpretar_precision_con_gemini_mejorada(precision_valor, pred_variables, dias_diff=0, fecha_objetivo=None, datos_historicos=None):
    """Interpretación contextualizada considerando distancia temporal y análisis estacional"""
    
    # Análisis adicional de contexto temporal
    contexto_temporal = ""
    if fecha_objetivo:
        mes = fecha_objetivo.month
        contexto_estacional = analizar_contexto_estacional(mes, datos_historicos)
        contexto_temporal = f"\n- Contexto estacional: {contexto_estacional}"
    
    # Análisis de variabilidad histórica
    variabilidad_info = ""
    if datos_historicos is not None:
        variabilidad_info = analizar_variabilidad_historica(pred_variables, datos_historicos)
    
    if dias_diff > 730:  # Más de 2 años
        contexto_temporal_base = "predicción a muy largo plazo (>2 años)"
        recomendacion_base = "Esta predicción utiliza patrones climáticos históricos y tendencias estacionales. Considérela como una estimación orientativa."
    elif dias_diff > 365:  # Más de 1 año
        contexto_temporal_base = "predicción a largo plazo (>1 año)"  
        recomendacion_base = "El modelo combina tendencias históricas con patrones estacionales para esta estimación."
    elif dias_diff > 90:  # Más de 3 meses
        contexto_temporal_base = "predicción a medio plazo"
        recomendacion_base = "Predicción basada en tendencias recientes y patrones estacionales."
    else:
        contexto_temporal_base = "predicción a corto plazo"
        recomendacion_base = "Alta confiabilidad basada en datos recientes."
    
    # Ajustar mensaje según precisión y contexto temporal
    if precision_valor < 50:
        nivel_confianza = "🔴 <b>Baja confiabilidad</b>"
        accion = "Use como referencia general. Complementar con monitoreo local intensivo."
    elif precision_valor < 70:
        nivel_confianza = "🟡 <b>Confiabilidad moderada</b>"
        accion = "Útil para planificación preventiva. Mantener vigilancia."
    else:
        nivel_confianza = "🟢 <b>Alta confiabilidad</b>"
        accion = "Excelente para toma de decisiones preventivas."
    
    return f"""{nivel_confianza} ({precision_valor:.1f}%)

<b>Contexto:</b> {contexto_temporal_base}{contexto_temporal}
<b>Interpretación:</b> {recomendacion_base}

<b>Para La Gasca:</b> {accion}{variabilidad_info}

<b>Factores considerados:</b>
• Patrones estacionales históricos de Quito
• Tendencias climáticas de largo plazo
• Variabilidad específica del sector
• Ensemble de múltiples modelos predictivos

<b>Recomendación:</b> Dado el historial del aluvión de 2022, siempre complementar con observación directa de quebradas y condiciones locales."""

def analizar_contexto_estacional(mes, datos_historicos):
    """Analiza patrones estacionales para mejorar interpretación de precisión"""
    if datos_historicos is None:
        return "Datos estacionales no disponibles"
    
    try:
        # Filtrar datos del mismo mes en años anteriores
        datos_mes = datos_historicos[datos_historicos['fecha'].dt.month == mes]
        
        if len(datos_mes) > 0:
            precip_promedio = datos_mes['precipitacion_valor'].mean()
            precip_max = datos_mes['precipitacion_valor'].max()
            variabilidad = datos_mes['precipitacion_valor'].std()
            
            if mes in [1, 2, 3, 10, 11, 12]:  # Meses lluviosos
                return f"Época lluviosa (prom: {precip_promedio:.1f}mm, máx histórico: {precip_max:.1f}mm). Mayor variabilidad climática."
            else:
                return f"Época seca (prom: {precip_promedio:.1f}mm). Menor variabilidad, predicciones más estables."
        
        return "Datos estacionales insuficientes"
    except:
        return "Error al analizar datos estacionales"

def analizar_variabilidad_historica(pred_variables, datos_historicos):
    """Analiza la variabilidad histórica para contextualizar la precisión"""
    if datos_historicos is None:
        return ""
    
    try:
        precip_pred = pred_variables.get('precipitacion', 0)
        
        # Calcular percentiles históricos
        p25 = datos_historicos['precipitacion_valor'].quantile(0.25)
        p75 = datos_historicos['precipitacion_valor'].quantile(0.75)
        p95 = datos_historicos['precipitacion_valor'].quantile(0.95)
        
        contexto = ""
        if precip_pred <= p25:
            contexto = "\n- Precipitación predicha en rango bajo (25% inferior histórico) - Mayor confiabilidad esperada"
        elif precip_pred <= p75:
            contexto = "\n- Precipitación predicha en rango normal - Confiabilidad estándar"
        elif precip_pred <= p95:
            contexto = "\n- Precipitación predicha en rango alto (25% superior) - Requiere validación adicional"
        else:
            contexto = "\n- Precipitación predicha en rango extremo (5% superior) - Precisión incierta, MONITOREO CRÍTICO"
        
        return contexto
    except:
        return ""

def generar_interpretacion_precision_local_mejorada(precision_valor, pred_variables, dias_diff=0):
    """Fallback mejorado con análisis más detallado"""
    precip = pred_variables.get('precipitacion', 0)
    nivel = pred_variables.get('nivel_agua', 10)
    
    # Cálculo de riesgo combinado
    riesgo_precip = min(100, (precip / 70) * 100)  # 70mm umbral crítico
    riesgo_nivel = min(100, (nivel / 35) * 100)    # 35cm umbral estimado
    riesgo_combinado = (riesgo_precip + riesgo_nivel) / 2
    
    if precision_valor < 60:
        base_msg = f"🔴 <b>Precisión Baja ({precision_valor:.1f}%)</b>"
        if riesgo_combinado > 50:
            return f"""{base_msg}
            
<b>⚠️ SITUACIÓN CRÍTICA</b>: Baja precisión con riesgo elevado ({riesgo_combinado:.1f}%). 
<b>ACCIÓN INMEDIATA</b>: 
- Activar monitoreo manual cada 2 horas en quebradas
- Coordinar con ECU-911 para alertas tempranas
- Preparar evacuación preventiva si condiciones empeoran
- Validar con estaciones meteorológicas cercanas
<b>CONTEXTO LA GASCA</b>: Dado el historial de 2022, NO depender únicamente del modelo."""
        else:
            return f"""{base_msg}
            
<b>Condiciones</b>: Riesgo moderado-bajo pero precisión incierta.
<b>RECOMENDACIÓN</b>: Mantener vigilancia estándar pero complementar con:
- Observación visual de quebradas cada 6 horas
- Monitoreo de pronósticos oficiales (INAMHI)
- Comunicación activa con red comunitaria de La Gasca"""
    
    elif precision_valor < 80:
        base_msg = f"🟡 <b>Precisión Moderada ({precision_valor:.1f}%)</b>"
        if riesgo_combinado > 50:
            return f"""{base_msg}
            
<b>ALERTA PREVENTIVA</b>: Precisión aceptable con condiciones de riesgo.
<b>PLAN DE ACCIÓN</b>:
- Predicciones útiles para planificación de 12-24 horas
- Intensificar monitoreo comunitario en zonas vulnerables
- Revisar rutas de evacuación y comunicar a familias en riesgo
<b>CONFIABILIDAD</b>: Adecuada para decisiones preventivas graduales."""
        else:
            return f"""{base_msg}
            
<b>SITUACIÓN CONTROLADA</b>: Precisión y riesgo en niveles manejables.
<b>USO RECOMENDADO</b>: 
- Planificación de actividades comunitarias
- Preparación preventiva estándar
- Educación y simulacros de evacuación"""
    
    else:
        base_msg = f"🟢 <b>Precisión Alta ({precision_valor:.1f}%)</b>"
        return f"""{base_msg}
        
<b>EXCELENTE CONFIABILIDAD</b>: Predicciones altamente confiables para La Gasca.
<b>APLICACIÓN RECOMENDADA</b>:
- Planificación estratégica de emergencias
- Coordinación con autoridades municipales
- Base sólida para decisiones de evacuación preventiva
- Referencia confiable para protocolos comunitarios
<b>VALOR AGREGADO</b>: Permite anticipación efectiva en zona históricamente vulnerable."""


