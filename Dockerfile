# Informa a imagem base
FROM python:3.12

# Define o ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip install --upgrade pip 

# Define o diretório de trabalho onde o projeto irá rodar
WORKDIR /app

# Copia o arquivo de dependências para o diretório
COPY requirements.txt /app/

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto para o diretório
COPY . /app/ 

# Expõe a porta do Django
EXPOSE 8000
 
# Executa o servidor de desenvolvimento do Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]