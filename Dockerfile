FROM python:3.11-slim

WORKDIR /app

# Instala dependencias do sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o codigo
COPY app.py .
COPY index.html .

EXPOSE 5000

CMD ["python", "app.py"]
