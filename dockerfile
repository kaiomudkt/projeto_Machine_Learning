# Use a imagem base do Jupyter SciPy Notebook
FROM quay.io/jupyter/scipy-notebook:2024-03-14

USER root

# Instale o Tesseract OCR e o pacote de idioma inglês
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng

# Instale o pytesseract
RUN pip install pytesseract

USER jovyan
