from fastapi import FastAPI
import gradio as gr
from modelo import (
    entrenar_modelo,
    predecir_variables,
    crear_grafica,
    interpretar_con_gemini
)
import csv

# Diccionario de meses
meses_dict = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Iniciar FastAPI
app = FastAPI()

# Cargar modelos
modelos, df = entrenar_modelo()

# Función principal
def interfaz(mes: str, año: int):
    mes_num = meses_dict[mes]  # Convertir nombre del mes a número

    ultimos_valores = {
        'precipitacion_valor': df['precipitacion_valor'].iloc[-1],
        'temperatura_valor': df['temperatura_valor'].iloc[-1],
        'nivel_agua_valor': df['nivel_agua_valor'].iloc[-1],
        'presion_valor': df['presion_valor'].iloc[-1],
    }

    pred = predecir_variables(modelos, mes_num, año, ultimos_valores)

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
    graf_precip, exp_p = crear_grafica(modelos['precipitacion'], df, mes_num, año, 'precipitacion', color='blue')
    graf_nivel, exp_n = crear_grafica(modelos['nivel_agua'], df, mes_num, año, 'nivel_agua', color='teal')

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
        mes_input = gr.Dropdown(choices=list(meses_dict.keys()), label="📅 Mes de Predicción", value="Junio")
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
import os

# Archivos externos
PREGUNTAS_CSV = "educacion_preguntas.csv"
MAPA_IMG = "zona_lagasca_map.png"


# 1️⃣ Evaluación de riesgo
def evaluar_riesgo(precipitacion, pendiente, frecuencia, canales):
    if precipitacion > 50 and pendiente == "Alta" and canales == "Obstruidos":
        return "🚨 Riesgo extremo de desbordamiento. Alerta máxima."
    elif precipitacion > 30 and (pendiente == "Media" or frecuencia == "Alta frecuencia"):
        return "⚠️ Riesgo moderado. Revise estructuras y vías de evacuación."
    else:
        return "✅ Riesgo bajo. Mantenga vigilancia comunitaria activa."

# 2️⃣ Quiz de cultura climática
def verificar_respuesta(respuesta):
    if respuesta == "B) Riesgo de desbordamiento e inundación":
        return "✅ Correcto. Lluvias > 70 mm/h pueden generar desastres."
    else:
        return "❌ Incorrecto. Revisa la lección y vuelve a intentar."

# 3️⃣ Glosario con IA
def consultar_glosario(termino):
    from modelo import interpretar_variable
    return interpretar_variable(termino, 0)

# 4️⃣ Guardar consulta en CSV
def guardar_pregunta(texto):
    with open(PREGUNTAS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([texto])
    return "✅ Tu consulta ha sido enviada al equipo educativo."

# Pestaña completa
with gr.Blocks(title="📚 Educación Climática") as interfaz_educativa:
    # Explicación inicial (la que pediste mantener)
    gr.Markdown("## 📘 Educación Comunitaria")
    gr.Markdown(
        "📘 **¿Qué mide cada variable?**\n\n"
        "🌧️ **Precipitación:** Cantidad de lluvia diaria (mm). Ayuda a identificar eventos de riesgo por escorrentías.\n\n"
        "🌊 **Nivel de agua:** Altura del flujo en quebradas o canales. Se incrementa con lluvias fuertes o taponamientos.\n\n"
        "🎯 **Objetivo del sistema:** Prevenir desastres como los ocurridos el 31 de enero de 2022 mediante modelos predictivos."
    )

    with gr.Tab("1️⃣ Simulador de Riesgo"):
        gr.Markdown("🔍 Simula diferentes condiciones para evaluar el riesgo de desbordamiento.")

        lluvia = gr.Number(label="🌧️ Precipitación estimada (mm)", value=0)
        pendiente = gr.Radio(["Baja", "Media", "Alta"], label="⛰️ Pendiente del terreno")
        frecuencia = gr.Radio(["Esporádica", "Alta frecuencia"], label="⏰ Frecuencia de lluvias")
        canales = gr.Radio(["Limpios", "Obstruidos"], label="🧹 Estado de los canales")
        btn_simular = gr.Button("Evaluar riesgo")

        salida_simulador = gr.Textbox(label="🧠 Evaluación del Riesgo")
        btn_simular.click(evaluar_riesgo, [lluvia, pendiente, frecuencia, canales], salida_simulador)

    with gr.Tab("2️⃣ Mapa de Zonas Vulnerables"):
        gr.Markdown("🗺️ Visualiza sectores en La Gasca con antecedentes de desbordamientos.")
        if os.path.exists(MAPA_IMG):
            gr.Image(value=MAPA_IMG, label="Mapa de zonas críticas")
        else:
            gr.Markdown("⚠️ No se encontró el mapa. Asegúrate de que el archivo `zona_lagasca_mapa.png` esté en la carpeta del proyecto.")

    with gr.Tab("3️⃣ Quiz de Cultura Climática"):
        gr.Markdown("🎓 ¿Qué representa una lluvia mayor a 70 mm/h?")
        respuesta = gr.Radio(
            ["A) Buen momento para recolectar agua", 
             "B) Riesgo de desbordamiento e inundación", 
             "C) Día parcialmente nublado"],
            label="Selecciona una opción"
        )
        btn_verificar = gr.Button("Verificar")
        salida_quiz = gr.Textbox(label="📚 Resultado")
        btn_verificar.click(verificar_respuesta, respuesta, salida_quiz)

    with gr.Tab("4️⃣ Glosario climático con IA"):
        gr.Markdown("💬 Consulta palabras relacionadas al clima.")
        entrada_glosario = gr.Textbox(label="🔍 Término (ej: escorrentía, humedad, alerta)")
        btn_glosario = gr.Button("Consultar término")
        salida_glosario = gr.Textbox(label="📖 Definición")
        btn_glosario.click(consultar_glosario, entrada_glosario, salida_glosario)

    with gr.Tab("5️⃣ Deja tu pregunta"):
        gr.Markdown("📨 ¿Tienes dudas o sugerencias? Escríbelas aquí.")
        pregunta_usuario = gr.Textbox(label="✏️ Tu mensaje")
        btn_guardar = gr.Button("Enviar")
        salida_pregunta = gr.Textbox(label="📥 Estado")
        btn_guardar.click(guardar_pregunta, pregunta_usuario, salida_pregunta)


# Agrupar pestañas
gr_tabs = gr.TabbedInterface(
    interface_list=[interfaz_prediccion, interfaz_educativa],
    tab_names=["🌧️ Predicción", "📘 Educación"]
)

# Montar en FastAPI
app = gr.mount_gradio_app(app, gr_tabs, path="/gradio")
