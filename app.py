from fastapi import FastAPI
import gradio as gr
from modelo import entrenar_modelo, predecir_variables, crear_grafica, crear_grafica_precision, interpretar_con_gemini
import datetime
import csv
import os

meses_dict = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

app = FastAPI()
modelos, df = entrenar_modelo()

# ===============================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ===============================
def interfaz(mes: str, año: int, dia: int):
    mes_num = meses_dict[mes]
    año = int(año)
    dia = int(dia)
    ult = df.sort_values(by='fecha').iloc[-1]
    ultimos_valores = {
        'precipitacion_valor': ult['precipitacion_valor'],
        'temperatura_valor': ult['temperatura_valor'],
        'nivel_agua_valor': ult['nivel_agua_valor'],
        'presion_valor': ult['presion_valor'],
    }

    pred, precision, fechas_precision, valores_precision = predecir_variables(modelos, mes_num, año, dia, ultimos_valores, df)

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

    graf_precip, exp_p = crear_grafica(modelos['rf']['precipitacion'], df, mes_num, año, 'precipitacion', color='blue')
    graf_nivel, exp_n = crear_grafica(modelos['rf']['nivel_agua'], df, mes_num, año, 'nivel_agua', color='teal')
    graf_precision = crear_grafica_precision(fechas_precision, valores_precision)

    precision_str = f"📏 Precisión estimada del modelo: {round(precision, 1)}%"
    if precision < 60:
        precision_str += " 🔴"
    elif precision < 80:
        precision_str += " 🟡"
    else:
        precision_str += " 🟢"

    return texto_pred, texto_consejo, explicacion, graf_precip, exp_p, graf_nivel, exp_n, precision_str, graf_precision

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
        pred_output = gr.Textbox(label="📊 Resultados del Modelo", lines=3)
        consejo_output = gr.Textbox(label="🔐 Sugerencia de Prevención", lines=2)

    interpretacion_output = gr.Textbox(label="🤖 Interpretación Técnica con IA", lines=6, max_lines=10)

    with gr.Row():
        graf_precip = gr.Image(label="🌧️ Gráfica: Precipitación", type="pil")
        interpretacion_precip = gr.Textbox(label="📖 Interpretación Individual: Precipitación", lines=3,  max_lines=5)

    with gr.Row():
        graf_nivel = gr.Image(label="🌊 Gráfica: Nivel de Agua", type="pil")
        interpretacion_nivel = gr.Textbox(label="📖 Interpretación Individual: Nivel de Agua", lines=3,  max_lines=5)

    with gr.Row():
        precision_output = gr.Textbox(label="📏 Precisión del Modelo", lines=1)
        graf_precision = gr.Image(label="📉 Precisión proyectada", type="pil")

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

# ===============================
# PESTAÑA EDUCATIVA
# ===============================

PREGUNTAS_CSV = "educacion_preguntas.csv"
MAPA_IMG = "zona_lagasca_map.png"

def evaluar_riesgo(precipitacion, pendiente, frecuencia, canales):
    if precipitacion > 50 and pendiente == "Alta" and canales == "Obstruidos":
        return "🚨 Riesgo extremo de desbordamiento. Alerta máxima."
    elif precipitacion > 30 and (pendiente == "Media" or frecuencia == "Alta frecuencia"):
        return "⚠️ Riesgo moderado. Revise estructuras y vías de evacuación."
    else:
        return "✅ Riesgo bajo. Mantenga vigilancia comunitaria activa."

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
        lluvia = gr.Number(label="🌧️ Precipitación estimada (mm)", value=0)
        pendiente = gr.Radio(["Baja", "Media", "Alta"], label="⛰️ Pendiente del terreno")
        frecuencia = gr.Radio(["Esporádica", "Alta frecuencia"], label="⏰ Frecuencia de lluvias")
        canales = gr.Radio(["Limpios", "Obstruidos"], label="🧹 Estado de los canales")
        btn_simular = gr.Button("Evaluar riesgo")
        salida_simulador = gr.Textbox(label="🧠 Evaluación del Riesgo")
        btn_simular.click(evaluar_riesgo, [lluvia, pendiente, frecuencia, canales], salida_simulador)

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
