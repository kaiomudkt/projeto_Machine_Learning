# Use a imagem base do Jupyter SciPy Notebook
FROM quay.io/jupyter/scipy-notebook:2024-03-14

USER root

# Instale o Tesseract OCR e o pacote de idioma inglês
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng

# Copia os arquivos de requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER jovyan