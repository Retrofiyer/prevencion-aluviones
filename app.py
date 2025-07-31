from fastapi import FastAPI
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import datetime
import csv
from modelo import (
    entrenar_modelo,
    entrenar_modelo_mejorado,
    predecir_variables,
    predecir_variables_ensemble,
    crear_grafica,
    crear_grafica_precision,
    interpretar_con_gemini,  
    crear_grafica_lineal,
    crear_grafica_lineal_interactiva,
    crear_grafica_completa_interactiva,
    interpretar_variable,
    interpretar_precision_con_gemini_mejorada,
    generar_interpretacion_precision_local_mejorada,
    calcular_precision_temporal_mejorada
)

# Función para interpretar la precisión con contexto de La Gasca usando Gemini MEJORADA
def interpretar_precision_con_gemini(precision_valor, pred_variables, fecha_objetivo=None, datos_historicos=None):
    """
    Genera una interpretación contextualizada de la precisión del modelo
    usando análisis detallado de los datos históricos y patrones estacionales.
    """
    
    # Calcular distancia temporal si se proporciona fecha objetivo
    dias_diff = 0
    if fecha_objetivo and datos_historicos is not None:
        fecha_ultima = datos_historicos['fecha'].max()
        if isinstance(fecha_objetivo, str):
            fecha_objetivo = pd.to_datetime(fecha_objetivo)
        elif hasattr(fecha_objetivo, 'date'):
            fecha_objetivo = pd.Timestamp(fecha_objetivo)
        dias_diff = (fecha_objetivo - fecha_ultima).days
    
    # Usar la función mejorada de interpretación
    try:
        interpretacion = interpretar_precision_con_gemini_mejorada(
            precision_valor, pred_variables, dias_diff, fecha_objetivo, datos_historicos
        )
        return interpretacion
    except Exception as e:
        # Fallback mejorado con más contexto
        return generar_interpretacion_precision_local_mejorada(precision_valor, pred_variables, dias_diff)

app = FastAPI()

meses_dict = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Entrenamiento de modelos y carga de datos
modelos, df = entrenar_modelo()  # Modelo tradicional

# NUEVO: Entrenamiento del modelo mejorado para mejor precisión en fechas lejanas
try:
    modelos_ensemble, df_expandido, features_expandidas = entrenar_modelo_mejorado()
    print("✅ Modelo ensemble mejorado cargado exitosamente")
    USAR_MODELO_MEJORADO = True
except Exception as e:
    print(f"⚠️ Error al cargar modelo mejorado, usando modelo tradicional: {e}")
    modelos_ensemble = None
    df_expandido = df
    features_expandidas = None
    USAR_MODELO_MEJORADO = False

def mostrar_grafico_historico(variable):
    return crear_grafica_lineal(df, variable)

# Carga de datos históricos reales (usar el dataset limpio correcto)
historico_df = pd.read_csv("dataset_rumipamba_limpio.csv")
historico_df["fecha"] = pd.to_datetime(historico_df["fecha"])

# ===============================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN MEJORADA
# ===============================
def interfaz(mes: str, año: int, dia: int):
    mes_num = meses_dict[mes]
    fecha_pred = datetime.date(año, mes_num, dia)

    # Usar el dataset expandido si está disponible
    df_actual = df_expandido if USAR_MODELO_MEJORADO else df
    ult = df_actual.sort_values(by='fecha').iloc[-1]
    ultimos_valores = {
        'precipitacion_valor': ult['precipitacion_valor'],
        'temperatura_valor': ult['temperatura_valor'],
        'nivel_agua_valor': ult['nivel_agua_valor'],
        'presion_valor': ult['presion_valor'],
    }

    # Usar modelo mejorado si está disponible, sino usar modelo tradicional
    if USAR_MODELO_MEJORADO and modelos_ensemble is not None:
        pred, intervalos_confianza, precision, fechas_precision, valores_precision = predecir_variables_ensemble(
            modelos_ensemble, mes_num, año, dia, ultimos_valores, df_expandido, features_expandidas
        )
        
        # Agregar información sobre intervalos de confianza
        intervalos_info = "\n\n📊 **Intervalos de Confianza (95%):**\n"
        for variable, valor in pred.items():
            var_names = {
                'precipitacion': ('🌧️ Precipitación', 'mm'),
                'nivel_agua': ('🌊 Nivel de agua', 'cm'),
                'temperatura': ('🌡️ Temperatura', '°C'),
                'presion': ('📉 Presión atmosférica', 'hPa')
            }
            
            nombre, unidad = var_names[variable]
            intervalo_inf, intervalo_sup = intervalos_confianza[variable]
            intervalos_info += f"{nombre}: {valor:.1f} {unidad} (rango: {intervalo_inf:.1f} - {intervalo_sup:.1f})\n"
        
        modelo_info = "🔬 **Modelo Ensemble Mejorado** - Mayor precisión para fechas lejanas"
    else:
        pred, precision, fechas_precision, valores_precision = predecir_variables(
            modelos, mes_num, año, dia, ultimos_valores, df
        )
        intervalos_info = ""
        modelo_info = "📊 **Modelo Tradicional**"

    # Generar recomendaciones detalladas específicas para La Gasca
    def generar_recomendacion_detallada(pred_variables, fecha_pred):
        precipitacion = pred_variables["precipitacion"]
        nivel_agua = pred_variables["nivel_agua"]
        temperatura = pred_variables["temperatura"]
        presion = pred_variables["presion"]
        
        # Determinar nivel de riesgo y recomendaciones específicas
        if precipitacion > 50 and nivel_agua > 20:
            nivel_riesgo = "🚨 RIESGO ALTO"
            recomendaciones = [
                "🚨 ALERTA MÁXIMA para el sector La Gasca",
                "� Manténgase informado a través de medios oficiales",
                "🏃‍♂️ Revise y practique rutas de evacuación hacia zonas altas",
                "👂 Esté atento al sonido de las quebradas (rumor anormal indica peligro)",
                "🚪 Tenga listo un kit de emergencia (documentos, agua, medicinas)",
                "👥 Coordine con vecinos para monitoreo conjunto de quebradas",
                "🚫 EVITE transitar cerca de quebradas Rumipamba y afluentes",
                "📞 Números de emergencia: 911 (Nacional), ECU 911"
            ]
            contexto_historico = "⚠️ Recuerde: El 31 de enero de 2022, condiciones similares causaron el aluvión en La Gasca."
            
        elif precipitacion > 30 or nivel_agua > 15:
            nivel_riesgo = "⚠️ RIESGO MODERADO"
            recomendaciones = [
                "⚠️ Precaución elevada para residentes de La Gasca",
                "👀 Inspeccione canales y alcantarillas de su propiedad",
                "🧹 Limpie hojas y desechos de canales de drenaje",
                "📍 Identifique puntos altos cercanos como refugio temporal",
                "👂 Manténgase atento a sonidos inusuales de quebradas",
                "💬 Informe a vecinos sobre condiciones de riesgo",
                "📱 Mantenga celular cargado y radio disponible",
                "🎒 Prepare kit básico de emergencia por precaución"
            ]
            contexto_historico = "📚 La zona tiene historial de vulnerabilidad durante lluvias intensas."
            
        elif precipitacion > 10 or nivel_agua > 12:
            nivel_riesgo = "🟡 RIESGO BAJO-MODERADO"
            recomendaciones = [
                "🟡 Vigilancia preventiva recomendada",
                "🔍 Revise estado de canales y drenajes locales",
                "🧹 Mantenga limpios los desagües de su propiedad",
                "📰 Siga pronósticos meteorológicos actualizados",
                "👥 Mantenga comunicación con vecinos",
                "📋 Verifique que tenga números de emergencia disponibles",
                "🎒 Considere tener documentos importantes en lugar seguro"
            ]
            contexto_historico = "📍 Monitoreo preventivo es clave en zonas como La Gasca."
            
        else:
            nivel_riesgo = "✅ RIESGO BAJO"
            recomendaciones = [
                "✅ Condiciones climáticas estables para La Gasca",
                "🔧 Aproveche para mantener sistemas de drenaje",
                "📚 Momento ideal para educarse sobre prevención de desastres",
                "👥 Participe en actividades comunitarias de preparación",
                "📋 Actualice su plan familiar de emergencia",
                "🎒 Revise y actualice kit de emergencia familiar",
                "📱 Manténgase informado sobre el clima local"
            ]
            contexto_historico = "🌟 Condiciones favorables para actividades preventivas y educativas."
        
        # Añadir recomendaciones específicas por condiciones meteorológicas
        recomendaciones_adicionales = []
        
        if temperatura < 10:
            recomendaciones_adicionales.append("🥶 Temperatura baja: Protéjase del frío y mantenga calefacción segura")
            
        if presion < 680:
            recomendaciones_adicionales.append("🌀 Presión baja: Posible cambio de tiempo, manténgase alerta")
            
        if presion > 690:
            recomendaciones_adicionales.append("☀️ Presión alta: Tiempo estable, buen momento para preparativos")
        
        # Construir mensaje final
        mensaje = f"{nivel_riesgo}\n\n"
        mensaje += f"📅 Predicción para {fecha_pred.strftime('%d de %B de %Y')}\n\n"
        mensaje += "🎯 **RECOMENDACIONES ESPECÍFICAS PARA LA GASCA:**\n"
        
        for i, rec in enumerate(recomendaciones, 1):
            mensaje += f"{i}. {rec}\n"
        
        if recomendaciones_adicionales:
            mensaje += f"\n🌡️ **CONDICIONES METEOROLÓGICAS ADICIONALES:**\n"
            for rec in recomendaciones_adicionales:
                mensaje += f"• {rec}\n"
        
        mensaje += f"\n📖 **CONTEXTO:** {contexto_historico}"
        
        return mensaje

    # Generar recomendación detallada
    consejo_detallado = generar_recomendacion_detallada(pred, fecha_pred)

    texto_pred = (
        f"🌧️ Precipitación estimada: {pred['precipitacion']} mm\n"
        f"🌊 Nivel de agua estimado: {pred['nivel_agua']} cm\n"
        f"🌡️ Temperatura estimada: {pred['temperatura']} °C\n"
        f"📉 Presión atmosférica estimada: {pred['presion']} hPa"
    )

    fila_real = historico_df[historico_df["fecha"] == pd.to_datetime(fecha_pred)]
    if not fila_real.empty:
        fila_real = fila_real.iloc[0]
        comparacion = []
        for key_pred, key_real, label, unidad in [
            ("precipitacion", "precipitacion_valor", "🌧️ Precipitación", "mm"),
            ("nivel_agua", "nivel_agua_valor", "🌊 Nivel de agua", "cm"),
            ("temperatura", "temperatura_valor", "🌡️ Temperatura", "°C"),
            ("presion", "presion_valor", "📉 Presión atmosférica", "hPa")
        ]:
            predicho = pred[key_pred]
            real = fila_real[key_real]
            diferencia = abs(predicho - real)
            error_pct = round(diferencia / real * 100, 2) if real != 0 else 0
            comparacion.append(
                f"{label}: Predicho {predicho} {unidad} vs Real {real} {unidad} → Diferencia: {error_pct}%"
            )
        
        # Agregar información sobre las fuentes de datos reales solo cuando hay comparación
        fuentes_info = (
            "\n\n📋 Fuentes de datos reales utilizados en la comparación:\n"
            "🌐 Visual Crossing Weather History (https://www.visualcrossing.com/weather-history/)\n"
            "🌐 World Weather Online - Quito Historical Data (https://www.worldweatheronline.com/quito-weather-history/pichincha/ec.aspx)\n"
            "📍 Datos correspondientes a la zona de Quito-Pichincha, Ecuador\n"
            "📅 Período de referencia: 2021-2024"
        )
        
        texto_pred += "\n\n📊 Comparación con datos reales:\n" + "\n".join(comparacion) + fuentes_info
    else:
        # Solo mensaje simple cuando no hay datos disponibles
        texto_pred += "\n\nℹ️ No hay datos reales disponibles para esta fecha."

    texto_consejo = consejo_detallado
    explicacion = interpretar_con_gemini(pred)

    # ✅ Gráficas interactivas con Plotly
    graf_precip = crear_grafica_completa_interactiva(
    modelos['rf']['precipitacion'], df, 'precipitacion', mes_num, año, color='blue'
    )

    graf_nivel = crear_grafica_completa_interactiva(
        modelos['rf']['nivel_agua'], df, 'nivel_agua', mes_num, año, color='teal'
    )

    graf_presion = crear_grafica_completa_interactiva(
        modelos['rf']['presion'], df, 'presion', mes_num, año, color='gray'
    )

    # Generar interpretaciones individuales
    interpretacion_precip = interpretar_variable('precipitacion', pred['precipitacion'])
    interpretacion_nivel = interpretar_variable('nivel_agua', pred['nivel_agua'])


    graf_precision = crear_grafica_precision(fechas_precision, valores_precision)

    # Generar interpretación contextualizada con Gemini para La Gasca
    precision_str = interpretar_precision_con_gemini(precision, pred)

    return (
        texto_pred,
        texto_consejo,
        explicacion,
        graf_precip,
        interpretacion_precip,
        graf_nivel,
        interpretacion_nivel,
        precision_str,
        graf_precision
    )
# ===============================
# PESTAÑA INICIO
# ===============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interfaz_inicio:

    # Hero section
    with gr.Row():
        gr.HTML("""
            <div style='text-align:center; padding:2rem; background:#B2DBF5; color:#0D47A1; border-radius:15px; box-shadow:0 4px 8px rgba(0,0,0,0.2); max-width:900px; margin:auto;'>
                <h1 style='font-size:3rem;'>🌧️ Estación Inteligente La Gasca</h1>
                <p style='font-size:1.5rem;'>Predicción y prevención de riesgos por lluvias intensas</p>
                <p style='max-width: 70%; margin: 1rem auto; color:#003366;'>
                    Plataforma de monitoreo climático basada en aprendizaje automático, diseñada para anticipar desbordamientos e inundaciones en zonas vulnerables.
                </p>
            </div>
        """)

    # Zona de Monitoreo
    with gr.Row():
        gr.HTML("""
            <div style='width:100%; max-width:900px; margin:auto; background:#E0F7FA; padding:1.5rem; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.05); text-align:center;'>
            <h3 style="text-align:left;">📍 Zona de Monitoreo: Sector La Gasca</h3>
            <p>📡 Esta plataforma utiliza datos reales obtenidos desde la estación <strong>P08 - Rumipamba Bodegas</strong>, descargados desde la red de monitoreo de 
            <a href="https://paramh2o.aguaquito.gob.ec/reportes/consultas/" target="_blank">PARAMH2O - Agua de Quito</a>.</p>
            <p>📍 Esta estación se encuentra cerca de <strong>La Gasca</strong>, zona afectada por el deslave del 31 de enero de 2022. Su ubicación estratégica permite captar condiciones atmosféricas y fluviales en tiempo real.</p>
            <p><strong>🔎 Variables monitoreadas:</strong></p>
            <ul>
                <li>🌧️ Precipitación (mm)</li>
                <li>🌡️ Temperatura (°C)</li>
                <li>🌊 Nivel de agua (cm)</li>
                <li>🌀 Presión atmosférica (hPa)</li>
            </ul>
            <p>📊 Estos datos alimentan un modelo predictivo basado en aprendizaje automático para detectar condiciones de alto riesgo y emitir recomendaciones preventivas.</p>
            
            <div style='background:#FFF9C4; padding:1rem; border-radius:8px; margin-top:1rem; border-left:4px solid #FBC02D;'>
                <h4 style='margin-top:0; color:#F57F17;'>📋 Fuentes de Datos de Validación</h4>
                <p style='margin-bottom:0; text-align:left; font-size:0.9rem;'>
                    🌐 <strong>Datos históricos de comparación obtenidos de:</strong><br>
                    • <a href="https://www.visualcrossing.com/weather-history/" target="_blank">Visual Crossing Weather History</a><br>
                    • <a href="https://www.worldweatheronline.com/quito-weather-history/pichincha/ec.aspx" target="_blank">World Weather Online - Quito Historical Data</a><br>
                    📅 <strong>Período:</strong> 2021-2024 | 📍 <strong>Zona:</strong> Quito-Pichincha, Ecuador
                </p>
            </div>
            </div>
        """)

    # Estación
    with gr.Row():
        gr.HTML("""
            <div style='width:100%; max-width:900px; margin:auto; background:#E0F7FA; padding:1.5rem; border-radius:12px; box-shadow:0 2px 6px rgba(0,0,0,0.05); text-align:center;'>
                <h3 style='margin-bottom:0.5rem;'>🛰️ Estación P08 - Rumipamba Bodegas</h3>
                <ul style='list-style:none; padding-left:0; line-height:1.8rem;'>
                    <li>📌 <strong>Ubicación:</strong> Entre el Parque Inglés y la quebrada Rumipamba</li>
                    <li>📍 <strong>Altitud:</strong> Aprox. 2,800 msnm</li>
                    <li>🧭 <strong>Coordenadas:</strong> -0.18106731817241586, -78.5099915241985</li>
                    <li>🔄 <strong>Frecuencia de registro:</strong> 5-15 minutos</li>
                    <li>💾 <strong>Historial utilizado:</strong> 2021–2024</li>
                </ul>
            </div>
        """)

    with gr.Row():
        gr.HTML("""
            <div style='width:100%; max-width:900px; margin:auto; border-radius:12px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                <iframe 
                    width="100%" 
                    height="400" 
                    frameborder="0" 
                    scrolling="no" 
                    marginheight="0" 
                    marginwidth="0"
                    src="https://www.openstreetmap.org/export/embed.html?bbox=-78.5120%2C-0.1831%2C-78.5080%2C-0.1791&layer=mapnik&marker=-0.18106731817241586%2C-78.5099915241985" 
                    style="border:1px solid #ccc;">
                </iframe>
                <div style="text-align:center; margin-top:0.5rem;">
                    <a href="https://www.openstreetmap.org/?mlat=-0.18106731817241586&mlon=-78.5099915241985#map=17/-0.18106731817241586/-78.5099915241985" target="_blank" style="color:#0D47A1;">
                        📍 Ver ubicación en OpenStreetMap
                    </a>
                </div>
            </div>
        """)
    # Objetivos
    with gr.Row():
        gr.HTML("<h3 style='text-align:center; width:100%;'>🎯 Objetivos de Investigación</h3>")

    with gr.Row():
        gr.HTML("""
            <div style='background:#F6D7E0; padding:1.5rem; border-radius:12px; width:45%; margin:auto; color:#6A1B9A; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;'>
                <h3>🔍 Minería de Datos</h3>
                <p>Exploración profunda de series históricas de precipitación, presión y nivel de agua.</p>
            </div>
        """),
        gr.HTML("""
            <div style='background:#D0F7EF; padding:1.5rem; border-radius:12px; width:45%; margin:auto; color:#004D40; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;'>
                <h3>📈 Predicción Automatizada</h3>
                <p>Modelos de aprendizaje automático para estimar variables críticas y activar alertas tempranas.</p>
            </div>
        """)

    # Footer
    gr.HTML("""
        <footer style='text-align:center; padding:1rem; font-size:0.85rem; color:#777; background:#f0f0f0; border-radius:0 0 15px 15px; margin-top:1rem;'>
            © 2025 Estación Inteligente - Proyecto Universitario de Monitoreo Climático en Quito, Ecuador.
        </footer>
    """)


# ===============================
# PESTAÑA PREDICCIÓN
# ===============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interfaz_prediccion:
    gr.Markdown("## 🌧️ Predicción de Riesgo por Lluvias - La Gasca")
    gr.Markdown(
        "👋 Bienvenido a la herramienta de predicción climática comunitaria para el sector **La Gasca** en Quito.\n\n"
        "📍 Esta app permite anticipar eventos de riesgo climático relacionados con **lluvias intensas y desbordamientos**.\n"
        "✅ Basado en datos reales de la estación Rumipamba."
    )

    with gr.Row():
        mes_input = gr.Dropdown(choices=list(meses_dict.keys()), label="📅 Mes de Predicción", value="Junio")
        año_input = gr.Number(label="🗓️ Año de Predicción", value=2026)
        dia_input = gr.Number(label="📆 Día de Predicción", value=1)
        btn = gr.Button("🔍 Evaluar Riesgo")

    with gr.Row():
        with gr.Column(scale=1):
            pred_output = gr.Textbox(label="📊 Resultados del Modelo", lines=6, max_lines=12, show_copy_button=True)
        with gr.Column(scale=1):
            consejo_output = gr.Textbox(
                label="�️ Recomendaciones de Prevención - La Gasca", 
                lines=12, 
                max_lines=20, 
                show_copy_button=True,
                interactive=False
            )

    interpretacion_output = gr.HTML(label="🤖 Interpretación Técnica con IA")

    # Gráfica de Precipitación - Layout mejorado
    with gr.Row():
        with gr.Column(scale=7):  # 70% del ancho para la gráfica
            graf_precip = gr.Plot(label="🌧️ Gráfica: Precipitación")
        with gr.Column(scale=3):  # 30% del ancho para la interpretación
            interpretacion_precip = gr.HTML(
                label="📖 Interpretación Individual: Precipitación"
            )

    # Gráfica de Nivel de Agua - Layout mejorado
    with gr.Row():
        with gr.Column(scale=7):  # 70% del ancho para la gráfica
            graf_nivel = gr.Plot(label="🌊 Gráfica: Nivel de Agua")
        with gr.Column(scale=3):  # 30% del ancho para la interpretación
            interpretacion_nivel = gr.HTML(
                label="📖 Interpretación Individual: Nivel de Agua"
            )
        
    with gr.Row():
        with gr.Column(scale=2):
            graf_precision = gr.Image(label="📊 Precisión proyectada", type="pil")
        with gr.Column(scale=1):
            precision_output = gr.HTML(
                label="🎯 Interpretación de Precisión - Contexto La Gasca"
            )

    btn.click(
        fn=interfaz,
        inputs=[mes_input, año_input, dia_input],
        outputs=[
            pred_output,
            consejo_output,
            interpretacion_output,
            graf_precip, interpretacion_precip,
            graf_nivel, interpretacion_nivel,
            precision_output,
            graf_precision
        ]
    )
    
#     gr.Markdown("## 📉 Evolución histórica de las variables")
    
# # Botón y selector en una fila
#     with gr.Row():
#         select_variable_historico = gr.Dropdown(
#             choices=[
#                 "precipitacion_valor", 
#                 "temperatura_valor", 
#                 "nivel_agua_valor", 
#                 "presion_valor"
#             ],
#             label="Selecciona variable a visualizar",
#             value="precipitacion_valor"
#         )
#         btn_ver_grafico = gr.Button("📊 Ver gráfico histórico")

#     # El gráfico solo, en una fila aparte
#     with gr.Row():
#         grafico_historico_output = gr.Plot(label="📈 Gráfico histórico interactivo", elem_id="grafico_historico")
    
#     from modelo import crear_grafica_lineal_interactiva

#     btn_ver_grafico.click(
#         fn=lambda variable: crear_grafica_lineal_interactiva(df, variable),
#         inputs=select_variable_historico,
#         outputs=grafico_historico_output
#     )


# ===============================
# PESTAÑA EDUCATIVA
# ===============================

PREGUNTAS_CSV = "educacion_preguntas.csv"
MAPA_IMG = "zona_lagasca_map.png"

def evaluar_riesgo(precipitacion, pendiente, frecuencia, canales):
    """
    Evaluación detallada de riesgo con consecuencias específicas
    basadas en las condiciones simuladas para La Gasca
    """
    
    # Calcular índice de riesgo ponderado
    puntaje_riesgo = 0
    factores_criticos = []
    consecuencias_especificas = []
    acciones_inmediatas = []
    
    # Análisis de precipitación
    if precipitacion >= 70:
        puntaje_riesgo += 40
        factores_criticos.append(f"Precipitación CRÍTICA ({precipitacion}mm)")
        consecuencias_especificas.append("🌊 SATURACIÓN TOTAL DEL SUELO: El agua no se infiltra, todo fluye hacia quebradas")
        consecuencias_especificas.append("⚡ CRECIMIENTO EXPONENCIAL DE CAUDALES: Nivel de quebradas puede triplicarse en 30 minutos")
        acciones_inmediatas.append("📞 LLAMAR 911 INMEDIATAMENTE - Reportar situación crítica")
        acciones_inmediatas.append("🏃 EVACUAR PREVENTIVAMENTE - Especialmente adultos mayores y niños")
    elif precipitacion >= 50:
        puntaje_riesgo += 30
        factores_criticos.append(f"Precipitación ALTA ({precipitacion}mm)")
        consecuencias_especificas.append("💧 SATURACIÓN PROGRESIVA: Suelo perdiendo capacidad de absorción en 2-4 horas")
        consecuencias_especificas.append("📈 CRECIMIENTO SOSTENIDO DE CAUDALES: Quebradas subiendo gradualmente pero constante")
        acciones_inmediatas.append("👁️ VIGILANCIA INTENSIVA - Monitoreo cada 30 minutos en quebradas")
        acciones_inmediatas.append("🎒 PREPARAR KIT DE EMERGENCIA - Documentos, medicinas, agua, radio")
    elif precipitacion >= 30:
        puntaje_riesgo += 20
        factores_criticos.append(f"Precipitación MODERADA ({precipitacion}mm)")
        consecuencias_especificas.append("🟡 INCREMENTO GRADUAL DE FLUJOS: Quebradas comenzando a crecer")
        consecuencias_especificas.append("⚠️ REDUCCIÓN DE MARGEN DE SEGURIDAD: Menos tiempo para reaccionar si empeora")
        acciones_inmediatas.append("👀 MONITOREO CADA 2 HORAS - Observar cambios en quebradas")
        acciones_inmediatas.append("📱 COORDINAR CON VECINOS - Red de alerta comunitaria activa")
    elif precipitacion >= 10:
        puntaje_riesgo += 10
        factores_criticos.append(f"Precipitación LIGERA ({precipitacion}mm)")
        consecuencias_especificas.append("✅ CONDICIONES CONTROLADAS: Drenaje natural funcionando adecuadamente")
        acciones_inmediatas.append("🧹 APROVECHA PARA MANTENIMIENTO - Limpieza de canales y desagües")
    else:
        consecuencias_especificas.append("☀️ CONDICIONES SECAS: Momento ideal para preparación y prevención")
        acciones_inmediatas.append("📋 PLANIFICACIÓN PREVENTIVA - Actualizar rutas de evacuación")
    
    # Análisis de pendiente del terreno
    if pendiente == "Alta":
        puntaje_riesgo += 25
        factores_criticos.append("PENDIENTE CRÍTICA (>30°)")
        consecuencias_especificas.append("🏔️ VELOCIDAD EXTREMA DEL AGUA: Flujo hasta 3x más rápido que en terreno plano")
        consecuencias_especificas.append("🪨 ARRASTRE DE MATERIAL PESADO: Rocas, troncos y sedimentos como proyectiles")
        consecuencias_especificas.append("⚡ TIEMPO DE EVACUACIÓN CRÍTICO: Solo 10-15 minutos desde alerta hasta impacto")
    elif pendiente == "Media":
        puntaje_riesgo += 15
        factores_criticos.append("PENDIENTE MODERADA (15-30°)")
        consecuencias_especificas.append("🌊 FLUJO ACELERADO: Agua 50% más rápida que en zona plana")
        consecuencias_especificas.append("⏰ VENTANA DE EVACUACIÓN REDUCIDA: 30-45 minutos para actuar")
    else:  # Baja
        puntaje_riesgo += 5
        consecuencias_especificas.append("🛤️ FLUJO CONTROLADO: Pendiente permite drenaje gradual y evacuación segura")
    
    # Análisis de frecuencia de lluvias
    if frecuencia == "Alta frecuencia":
        puntaje_riesgo += 15
        factores_criticos.append("SATURACIÓN ACUMULATIVA")
        consecuencias_especificas.append("💧 SUELO PRE-SATURADO: Menos capacidad de absorción, todo fluye superficialmente")
        consecuencias_especificas.append("📊 EFECTO MULTIPLICADOR: Cada mm adicional tiene impacto 2x mayor")
        acciones_inmediatas.append("📈 MONITOREO ACUMULATIVO - Sumar precipitación de últimos 7 días")
    else:  # Esporádica
        puntaje_riesgo += 5
        consecuencias_especificas.append("🌱 SUELO CON CAPACIDAD ABSORBENTE: Primera lluvia en días, mejor infiltración")
    
    # Análisis crítico del estado de canales
    if canales == "Obstruidos":
        puntaje_riesgo += 20
        factores_criticos.append("DRENAJE COMPROMETIDO")
        consecuencias_especificas.append("🚫 EFECTO REPRESAMIENTO: Agua acumulándose hasta romper obstáculos súbitamente")
        consecuencias_especificas.append("💥 LIBERACIÓN SÚBITA: 'Tsunami local' cuando se rompen obstrucciones")
        consecuencias_especificas.append("🏠 INUNDACIÓN DE VIVIENDAS: Agua busca rutas alternativas, incluyendo calles y casas")
        acciones_inmediatas.append("🔧 DESOBSTRUCCIÓN URGENTE - Si es seguro, remover obstáculos antes de lluvia fuerte")
        acciones_inmediatas.append("🚨 ALEJAR VEHÍCULOS Y PERTENENCIAS - De zonas bajas y cercanas a canales")
    else:  # Limpios
        consecuencias_especificas.append("✅ DRENAJE ÓPTIMO: Canales funcionando a capacidad máxima")
    
    # Determinar nivel de riesgo final
    if puntaje_riesgo >= 80:
        nivel_riesgo = "🔴 RIESGO EXTREMO"
        urgencia = "⚡ EVACUACIÓN INMEDIATA ⚡"
        contexto_historico = f"Condiciones IGUALES O PEORES al aluvión del 31 enero 2022 que causó muertes en La Gasca."
    elif puntaje_riesgo >= 60:
        nivel_riesgo = "🚨 RIESGO CRÍTICO"
        urgencia = "🚨 ALERTA MÁXIMA - EVACUAR PREVENTIVAMENTE 🚨"
        contexto_historico = f"Condiciones similares al desastre de 2022. Riesgo real de aluvión."
    elif puntaje_riesgo >= 40:
        nivel_riesgo = "⚠️ RIESGO ALTO"
        urgencia = "⚠️ PREPARAR EVACUACIÓN - VIGILANCIA EXTREMA ⚠️"
        contexto_historico = f"Condiciones preocupantes. A pasos del umbral crítico de 2022."
    elif puntaje_riesgo >= 25:
        nivel_riesgo = "🟡 RIESGO MODERADO"
        urgencia = "🟡 VIGILANCIA ACTIVA - PREPARACIÓN PREVENTIVA 🟡"
        contexto_historico = f"Condiciones que requieren atención. Potencial de escalamiento rápido."
    else:
        nivel_riesgo = "✅ RIESGO BAJO"
        urgencia = "✅ SITUACIÓN CONTROLADA ✅"
        contexto_historico = f"Condiciones normales. Momento ideal para preparación y educación."
    
    # Generar reporte detallado
    reporte = f"""{nivel_riesgo} (Índice: {puntaje_riesgo}/100)

{urgencia}

🎯 FACTORES CRÍTICOS IDENTIFICADOS:
{' • '.join(factores_criticos) if factores_criticos else '• Condiciones dentro de parámetros normales'}

⚡ CONSECUENCIAS ESPECÍFICAS CON ESTAS CONDICIONES:
{''.join([f'• {cons}\n' for cons in consecuencias_especificas[:6]])}

📋 ACCIONES INMEDIATAS REQUERIDAS:
{''.join([f'• {acc}\n' for acc in acciones_inmediatas[:4]])}

📚 CONTEXTO HISTÓRICO LA GASCA:
{contexto_historico}

🔗 CADENA DE CONSECUENCIAS ESPERADA:
1️⃣ PRIMEROS 15 MIN: {precipitacion}mm lluvia + pendiente {pendiente.lower()} = {"flujo extremo" if puntaje_riesgo > 60 else "incremento de caudal"}
2️⃣ 30-60 MIN: Canales {"colapsarán por obstrucciones" if canales == "Obstruidos" and puntaje_riesgo > 40 else "manejarán el flujo"} + {"suelo saturado" if frecuencia == "Alta frecuencia" else "absorción gradual"}
3️⃣ 1-2 HORAS: {"ALUVIÓN PROBABLE - Impacto en viviendas" if puntaje_riesgo >= 60 else "Monitoreo continuo necesario" if puntaje_riesgo >= 25 else "Situación estabilizada"}

⏰ TIEMPO ESTIMADO PARA ACTUAR: {f"5-10 minutos (CRÍTICO)" if puntaje_riesgo >= 80 else f"15-30 minutos" if puntaje_riesgo >= 60 else f"1-2 horas" if puntaje_riesgo >= 40 else "Varias horas disponibles"}"""
    
    return reporte

def verificar_respuesta(respuesta):
    return "✅ Correcto. Lluvias > 70 mm/h pueden generar desastres." if respuesta == "B) Riesgo de desbordamiento e inundación" else "❌ Incorrecto. Revisa la lección y vuelve a intentar."

def consultar_glosario(termino):
    from modelo import interpretar_variable
    return interpretar_variable(termino, 0)

def guardar_pregunta(texto):
    with open(PREGUNTAS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([texto])
    return "✅ Tu consulta ha sido enviada al equipo educativo."

with gr.Blocks(title="📚 Educación Climática") as interfaz_educativa:
    gr.Markdown("## 📘 Educación Comunitaria")
    gr.Markdown(
        "📘 **¿Qué mide cada variable?**\n\n"
        "🌧️ **Precipitación:** Cantidad de lluvia diaria (mm).\n"
        "🌊 **Nivel de agua:** Altura del flujo en quebradas.\n"
        "🎯 **Objetivo:** Prevenir desastres como el del 31 de enero de 2022."
    )

    with gr.Tab("1️⃣ Simulador de Riesgo"):
        gr.Markdown("""
        ### 🎯 **Simulador Avanzado de Riesgo de Aluviones - La Gasca**
        
        **Instrucciones:** Ajusta los parámetros según las condiciones actuales o hipotéticas para obtener una evaluación detallada de riesgo con consecuencias específicas y acciones recomendadas.
        
        ⚠️ **Referencia histórica:** El 31 de enero de 2022, lluvias >70mm/h causaron un aluvión devastador en La Gasca.
        """)
        
        with gr.Row():
            with gr.Column():
                lluvia = gr.Number(
                    label="🌧️ Precipitación estimada (mm/hora)", 
                    value=0, 
                    minimum=0, 
                    maximum=150,
                    info="0-10: Ligera | 10-30: Moderada | 30-50: Intensa | 50-70: Fuerte | >70: CRÍTICA"
                )
                pendiente = gr.Radio(
                    ["Baja", "Media", "Alta"], 
                    label="⛰️ Pendiente del terreno",
                    info="Baja: <15° | Media: 15-30° | Alta: >30°"
                )
            with gr.Column():
                frecuencia = gr.Radio(
                    ["Esporádica", "Alta frecuencia"], 
                    label="⏰ Frecuencia de lluvias recientes",
                    info="¿Ha llovido intensamente en los últimos 7 días?"
                )
                canales = gr.Radio(
                    ["Limpios", "Obstruidos"], 
                    label="🧹 Estado de canales y quebradas",
                    info="Presencia de escombros, basura, sedimentos"
                )
        
        btn_simular = gr.Button("🧠 EVALUAR RIESGO INTEGRAL", variant="primary", scale=2)
        
        gr.Markdown("### 📊 **Reporte Detallado de Evaluación:**")
        salida_simulador = gr.Textbox(
            label="", 
            lines=20,
            max_lines=25,
            show_label=False,
            placeholder="Selecciona los parámetros arriba y presiona 'EVALUAR RIESGO INTEGRAL' para obtener un análisis detallado...",
            interactive=False
        )
        
        btn_simular.click(evaluar_riesgo, [lluvia, pendiente, frecuencia, canales], salida_simulador)
        
        gr.Markdown("""
        ---
        **💡 Tip educativo:** Usa valores reales observados en tu zona para entrenar tu capacidad de evaluación de riesgo. 
        
        **🚨 Recordatorio:** En caso de condiciones críticas reales, llamar inmediatamente al 911.
        """)

    with gr.Tab("2️⃣ Quiz de Cultura Climática"):

        # Preguntas y respuestas del quiz
        preguntas_quiz = [
            {
                "pregunta": "¿Qué puede ocurrir si las lluvias superan los 70 mm por hora?",
                "opciones": [
                    "A) Buen momento para pasear",
                    "B) Riesgo de desbordamiento e inundación",
                    "C) Día parcialmente nublado"
                ],
                "respuesta_correcta": "B) Riesgo de desbordamiento e inundación"
            },
            {
                "pregunta": "¿Qué indica un aumento súbito en el nivel de agua de una quebrada?",
                "opciones": [
                    "A) Que es buen momento para bañarse",
                    "B) Que puede haber un aluvión",
                    "C) Que la presión atmosférica está estable"
                ],
                "respuesta_correcta": "B) Que puede haber un aluvión"
            },
            {
                "pregunta": "¿Qué acción es adecuada en caso de lluvias intensas y canales obstruidos?",
                "opciones": [
                    "A) Mantenerse en casa y no revisar nada",
                    "B) Ignorar el pronóstico",
                    "C) Verificar rutas de evacuación y alertar a vecinos"
                ],
                "respuesta_correcta": "C) Verificar rutas de evacuación y alertar a vecinos"
            },
            {
                "pregunta": "¿Cuál es una señal de alerta temprana en una quebrada?",
                "opciones": [
                    "A) Ruido de piedras arrastradas",
                    "B) Cielo despejado",
                    "C) Temperatura baja"
                ],
                "respuesta_correcta": "A) Ruido de piedras arrastradas"
            },
            {
                "pregunta": "¿Qué mide la variable 'precipitación'?",
                "opciones": [
                    "A) Cantidad de vapor en el aire",
                    "B) Cantidad de luz solar",
                    "C) Cantidad de lluvia en un período"
                ],
                "respuesta_correcta": "C) Cantidad de lluvia en un período"
            }
        ]

        # Estados del quiz
        indice_pregunta = gr.State(value=0)
        puntaje = gr.State(value=0)
        texto_pregunta = gr.Markdown()
        opciones_radio = gr.Radio(choices=[], label="Selecciona una opción", interactive=True)
        resultado_texto = gr.Textbox(label="📚 Resultado", lines=2)
        puntaje_texto = gr.Textbox(label="🎯 Puntaje", value="Puntaje: 0/5")
        boton_verificar = gr.Button("✅ Verificar")
        boton_siguiente = gr.Button("Siguiente pregunta ▶️")

        # Función para mostrar pregunta y opciones
        def mostrar_pregunta(index):
            if index < len(preguntas_quiz):
                pregunta = preguntas_quiz[index]
                return (
                    f"**Pregunta {index+1} de {len(preguntas_quiz)}**\n\n{pregunta['pregunta']}",
                    gr.update(choices=pregunta["opciones"], value=None),
                    "",
                    gr.update(visible=False)  # Ocultar botón siguiente
                )
            else:
                return (
                    "✅ ¡Has completado el quiz! Gracias por participar.",
                    gr.update(choices=[], value=None, interactive=False),
                    "🎉 Finalizado",
                    gr.update(visible=False)
                )

        # Función para verificar la respuesta
        def verificar_respuesta(respuesta, index, puntaje_actual):
            if index >= len(preguntas_quiz):
                return [
                    gr.update(interactive=False),
                    gr.update(),
                    "🎉 Quiz completado",
                    gr.update(visible=False),
                    index,
                    puntaje_actual,
                    f"Puntaje final: {puntaje_actual}/{len(preguntas_quiz)}"
                ]

            correcta = preguntas_quiz[index]["respuesta_correcta"]
            nuevo_puntaje = puntaje_actual
            if respuesta == correcta:
                resultado = "✅ ¡Correcto! Buena observación."
                nuevo_puntaje += 1
            else:
                resultado = f"❌ Incorrecto. La respuesta correcta era: {correcta}"

            return [
                gr.update(interactive=False),
                gr.update(value=f"**Pregunta {index+1} de {len(preguntas_quiz)}**\n\n{preguntas_quiz[index]['pregunta']}"),
                resultado,
                gr.update(visible=True),
                index,
                nuevo_puntaje,
                f"Puntaje: {nuevo_puntaje}/{len(preguntas_quiz)}"
            ]

        # Función para pasar a la siguiente pregunta
        def siguiente(index):
            next_index = index + 1
            if next_index >= len(preguntas_quiz):
                return (
                    "✅ ¡Has completado el quiz! Gracias por participar.",
                    gr.update(choices=[], value=None, interactive=False),
                    "🎉 Quiz completado",
                    gr.update(visible=False),
                    next_index
                )
            
            next_pregunta = preguntas_quiz[next_index]
            return (
                f"**Pregunta {next_index+1} de {len(preguntas_quiz)}**\n\n{next_pregunta['pregunta']}",
                gr.update(choices=next_pregunta["opciones"], value=None, interactive=True),
                "",
                gr.update(visible=False),
                next_index
            )

        # Cargar la primera pregunta al abrir la pestaña
        interfaz_educativa.load(
            fn=mostrar_pregunta,
            inputs=[indice_pregunta],
            outputs=[texto_pregunta, opciones_radio, resultado_texto, boton_siguiente]
        )

        # Al hacer clic en verificar
        boton_verificar.click(
            fn=verificar_respuesta,
            inputs=[opciones_radio, indice_pregunta, puntaje],
            outputs=[
                opciones_radio,
                texto_pregunta,
                resultado_texto,
                boton_siguiente,
                indice_pregunta,
                puntaje,
                puntaje_texto
            ]
        )
        
        # Al hacer clic en siguiente
        boton_siguiente.click(
            fn=siguiente,
            inputs=[indice_pregunta],
            outputs=[
                texto_pregunta,
                opciones_radio,
                resultado_texto,
                boton_siguiente,
                indice_pregunta
            ]
        )

    with gr.Tab("3️⃣ Glosario climático con IA"):
        entrada_glosario = gr.Textbox(label="🔍 Término")
        btn_glosario = gr.Button("Consultar término")
        salida_glosario = gr.Textbox(label="📖 Definición")
        btn_glosario.click(consultar_glosario, entrada_glosario, salida_glosario)

    with gr.Tab("4️⃣ Deja tu pregunta"):
        pregunta_usuario = gr.Textbox(label="✏️ Tu mensaje")
        btn_guardar = gr.Button("Enviar")
        salida_pregunta = gr.Textbox(label="📥 Estado")
        btn_guardar.click(guardar_pregunta, pregunta_usuario, salida_pregunta)

# ===============================
# PESTAÑA: ACERCA DE LA GASCA
# ===============================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interfaz_acerca:
    gr.Markdown("## 🗺️ Acerca de La Gasca")
    gr.Markdown(
        "📍 **La Gasca** es un barrio ubicado en las laderas del Pichincha, en Quito, Ecuador.\n\n"
        "🧱 El 31 de enero de 2022 ocurrió un desbordamiento de quebradas obstruidas por lluvias intensas, "
        "causando un desastre urbano con pérdidas humanas y materiales.\n\n"
        "🔬 Esta aplicación fue creada como respuesta preventiva basada en ciencia de datos y participación comunitaria."
    )

    # Mapa del deslave
    with gr.Row():
        gr.HTML("""
            <div style='width:100%; max-width:900px; margin:auto; border-radius:12px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
                <iframe 
                    width="100%" 
                    height="400" 
                    frameborder="0" 
                    scrolling="no" 
                    marginheight="0" 
                    marginwidth="0"
                    src="https://www.openstreetmap.org/export/embed.html?bbox=-78.5132%2C-0.1964%2C-78.5092%2C-0.1924&layer=mapnik&marker=-0.19437809656999522%2C-78.51118000811563" 
                    style="border:1px solid #ccc;">
                </iframe>
                <div style="text-align:center; margin-top:0.5rem;">
                    <p style="color:#e63946; font-weight:bold;">📍 Punto exacto del aluvión del 31 de enero de 2022</p>
                    <a href="https://www.openstreetmap.org/?mlat=-0.19437809656999522&mlon=-78.51118000811563#map=17/-0.19437809656999522/-78.51118000811563" 
                       target="_blank" style="color:#0D47A1;">
                        🔍 Ver en OpenStreetMap
                    </a>
                </div>
            </div>
        """)

    # Carrusel de fotos del deslave
    gr.Markdown("### 📸 Fotografías del Aluvión - 31 de enero de 2022")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("""
                <style>
                    #gallery-container {
                        width: 100%;
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    #gallery {
                        display: block;
                        margin: 0 auto;
                    }
                    #gallery img {
                        max-height: 500px;
                        object-fit: contain;
                        margin: 0 auto;
                        display: block;
                    }
                </style>
            """)
            carousel = gr.Gallery(
                [
                    "https://laboratoriolasa.com/wp-content/uploads/2025/06/aluvion-la-gasca-lab-lasa.webp",
                    "https://www.eltelegrafo.com.ec/media/k2/items/cache/1a1231f46d0ea9e70fb9c1a740c6d88f_XL.jpg",
                    "https://imagenes.primicias.ec/files/og_thumbnail/uploads/2024/05/26/6653af2ee9fb7.jpeg",
                    "https://www.ecuavisa.com/binrepository/1000x500/156c0/688d500/none/11705/YCMP/whatsapp-image-2022-02-03-at-12-29_318129_20220203142037.jpg"
                ],
                label="Registro fotográfico del aluvión",
                show_label=True,
                elem_id="gallery",
                columns=1,
                rows=1,
                height=500,
                object_fit="contain",
                allow_preview=True,
                show_download_button=False,
                container=True,
                preview=True
            )
    
    gr.Markdown(
        "> _Las imágenes muestran la magnitud del desastre ocurrido en el sector de La Gasca, donde el desbordamiento " 
        "de las quebradas causó severos daños a la infraestructura y afectó significativamente a la comunidad local._"
    )

# ===============================
# INTEGRACIÓN FINAL
# ===============================

gr_tabs = gr.TabbedInterface(
    interface_list=[interfaz_inicio, interfaz_prediccion, interfaz_educativa, interfaz_acerca],
    tab_names=["🏠 Inicio", "🌧️ Predicción", "📘 Educación", "🗺️ La Gasca"]
)

app = gr.mount_gradio_app(app, gr_tabs, path="/gradio")
