# Imagen base oficial de Python
FROM python:3.12.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivo de aplicación
COPY app.py .

# Comando para ejecutar el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]