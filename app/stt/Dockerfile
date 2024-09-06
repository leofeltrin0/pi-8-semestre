# Use a imagem base PyTorch com suporte para CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de requisitos e instale as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação
COPY . /app

# Exponha a porta para o FastAPI
EXPOSE 8000

# Comando para iniciar o servidor FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
