FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Указываем порт, который будет использовать Render
EXPOSE 7860

# Команда запуска
CMD ["python", "app.py"]