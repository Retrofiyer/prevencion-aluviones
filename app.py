from fastapi import FastAPI
import gradio as gr
from modelo import (
    entrenar_modelo,
    predecir_variables,
    crear_grafica,
    interpretar_con_gemini
)

# Iniciar FastAPI
app = FastAPI()

# Cargar modelos
modelos, df = entrenar_modelo()

# Función principal
def interfaz(mes: int, año: int):
    ultimos_valores = {
        'precipitacion_valor': df['precipitacion_valor'].iloc[-1],
        'temperatura_valor': df['temperatura_valor'].iloc[-1],
        'nivel_agua_valor': df['nivel_agua_valor'].iloc[-1],
        'presion_valor': df['presion_valor'].iloc[-1],
    }

    pred = predecir_variables(modelos, mes, año, ultimos_valores)

    if pred["precipitacion"] > 50 and pred["nivel_agua"] > 20:
        consejo = "🚨 Riesgo alto de desbordamiento. Manténgase alerta y revise rutas de evacuación."
    elif pred["precipitacion"] > 30:
        consejo = "⚠️ Lluvia moderada. Verifique canales y quebradas."
    else:
        consejo = "✅ Condiciones climáticas estables. Bajo riesgo de desbordamiento."

    texto_pred = (
        f"🌧️ Precipitación estimada: {pred['precipitacion']} mm\n"
        f"🌊 Nivel de agua estimado: {pred['nivel_agua']} cm"
    )
    texto_consejo = f"🛑 Recomendación: {consejo}"
    explicacion = interpretar_con_gemini(pred)

    # Gráficas individuales
    graf_precip, exp_p = crear_grafica(modelos['precipitacion'], df, mes, año, 'precipitacion', color='blue')
    graf_nivel, exp_n = crear_grafica(modelos['nivel_agua'], df, mes, año, 'nivel_agua', color='teal')

    return (
        texto_pred,
        texto_consejo,
        explicacion,
        graf_precip, exp_p,
        graf_nivel, exp_n
    )

# Crear interfaz principal con Blocks
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interfaz_prediccion:
    gr.Markdown("## 🌧️ Predicción de Riesgo por Lluvias - La Gasca")

    gr.Markdown(
        "👋 Bienvenido a la herramienta de predicción climática comunitaria para el sector **La Gasca** en Quito.\n\n"
        "📍 Esta app permite anticipar eventos de riesgo climático relacionados con **lluvias intensas y desbordamientos**.\n"
        "✅ Basado en datos reales de la estación Rumipamba."
    )

    with gr.Row():
        mes_input = gr.Slider(1, 12, step=1, label="📅 Mes de Predicción")
        año_input = gr.Number(label="🗓️ Año de Predicción", value=2026)
        btn = gr.Button("🔍 Evaluar Riesgo")

    with gr.Row():
        pred_output = gr.Textbox(label="📊 Resultados del Modelo", lines=3)
        consejo_output = gr.Textbox(label="🔐 Sugerencia de Prevención", lines=2)

    interpretacion_output = gr.Textbox(label="🤖 Interpretación Técnica con IA", lines=6)

    with gr.Row():
        graf_precip = gr.Image(label="🌧️ Gráfica: Precipitación", type="pil")
        interpretacion_precip = gr.Textbox(label="📖 Interpretación Individual: Precipitación", lines=2)

    gr.Markdown(
        "**🧾 Explicación:**\n"
        "🔵 Línea azul: datos históricos.\n"
        "🟠 Línea punteada: tendencia estimada.\n"
        "⚫ Línea gris: proyección mensual futura.\n"
        "🔴 Punto rojo: predicción exacta para el mes/año seleccionado."
    )

    with gr.Row():
        graf_nivel = gr.Image(label="🌊 Gráfica: Nivel de Agua", type="pil")
        interpretacion_nivel = gr.Textbox(label="📖 Interpretación Individual: Nivel de Agua", lines=2)

    gr.Markdown(
        "**🧾 Explicación:**\n"
        "🔵 Línea celeste: niveles históricos.\n"
        "🟠 Línea punteada: tendencia del modelo.\n"
        "⚫ Línea gris: proyección futura.\n"
        "🔴 Punto rojo: predicción del sistema."
    )

    # Acción del botón
    btn.click(
        fn=interfaz,
        inputs=[mes_input, año_input],
        outputs=[
            pred_output,
            consejo_output,
            interpretacion_output,
            graf_precip, interpretacion_precip,
            graf_nivel, interpretacion_nivel
        ]
    )

# Pestaña educativa
interfaz_educativa = gr.Interface(
    fn=lambda: (
        "📘 **¿Qué mide cada variable?**\n\n"
        "🌧️ **Precipitación:** Cantidad de lluvia diaria (mm). Ayuda a identificar eventos de riesgo por escorrentías.\n\n"
        "🌊 **Nivel de agua:** Altura del flujo en quebradas o canales. Se incrementa con lluvias fuertes o taponamientos.\n\n"
        "🎯 **Objetivo del sistema:** Prevenir desastres como los ocurridos el 31 de enero de 2022 mediante modelos predictivos."
    ),
    inputs=[],
    outputs=gr.Markdown(),
    title="📚 Educación Comunitaria",
    description="Información accesible para entender y prevenir riesgos hidrometeorológicos en tu barrio."
)

# Agrupar pestañas
gr_tabs = gr.TabbedInterface(
    interface_list=[interfaz_prediccion, interfaz_educativa],
    tab_names=["🌧️ Predicción", "📘 Educación"]
)

# Montar en FastAPI
app = gr.mount_gradio_app(app, gr_tabs, path="/gradio")