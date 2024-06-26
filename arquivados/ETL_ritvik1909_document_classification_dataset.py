# carrega dataset com_datasets_ritvik1909_document-classification-dataset e gera Dataframe pandas

import pandas as pd
import os
from typing import List
# NLTK (Natural Language Toolkit) 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer 
import re
from nltk.corpus import stopwords
# conjunto de palavras stopwords para o idioma inglês 
# Lista de palavras que são consideradas stopwords (palavras vazias) para o idioma inglês. 
stopwords_list = stopwords.words("english")
# stopwords_list = set(stopwords.words('english'))
from PIL import Image
import pytesseract
# Download dos recursos necessários do NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def get_class_labels():
    """
    Definição de Classes, opções de target
    """
    class_labels = { 'email':0, 'resume':1, 'scientific_publication':2 }
    return class_labels

def lowercase_and_remove_special_characters(text: str) -> str:
    """
    Converte o texto para minúsculas e remove caracteres especiais.
    - Conversão para Minúsculas 
    - Substituição de Quebras de Linha e Tabulações 
    - Remoção de Espaços Extras 
    - Remoção de Números
    - Remoção de Pontuação 
    Args:
        text (str): O texto original a ser processado.
    Returns:
        str: O texto processado com minúsculas e sem caracteres especiais.
    """
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub("\s+", " ", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_text(text: str) -> List[str]:
    """
    Tokeniza o texto usando NLTK word_tokenize.
    Args:
        text (str): O texto a ser tokenizado.
    Returns:
        List[str]: Lista de tokens gerada a partir do texto.
    """
    tokens = word_tokenize(text)
    return tokens

def remove_punctuation_and_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove pontuações e stopwords da lista de tokens.
    Args:
        tokens (List[str]): Lista de tokens a serem processados.
    Returns:
        List[str]: Lista de tokens sem pontuações e stopwords.
    """
    tokens = [token for token in tokens if token not in punctuation and token not in stopwords_list]
    return tokens

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Realiza a lematização dos tokens usando WordNetLemmatizer do NLTK.
    Args:
        tokens (List[str]): Lista de tokens a serem lematizados.
    Returns:
        List[str]: Lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def preprocess_text(text: str) -> str:
    """
    Realiza o pré-processamento completo do texto, incluindo minúsculas,
    remoção de caracteres especiais, tokenização, remoção de pontuações
    e stopwords, e lematização.
    Args:
        text (str): O texto original a ser pré-processado.
    Returns:
        str: O texto pré-processado final, pronto para uso em análises adicionais.
    """
    # Aplica as funções em sequência
    text = lowercase_and_remove_special_characters(text)
    tokens = tokenize_text(text)
    tokens = remove_punctuation_and_stopwords(tokens)
    final_text = lemmatize_tokens(tokens)
    return " ".join(final_text)

def process_image(label_path: str, file_name: str) -> str:
    """
    - carrega uma imagem.png
    - extrai o texto da imagem via OCR
    - realiza o pre processamento do texto
    retorna o texto ja pre processado
    """
    # Carrega a imagem
    image = Image.open(os.path.join(label_path, file_name))
    # Usa Tesseract OCR para extrair o texto da imagem
    text = pytesseract.image_to_string(image)
    # Executa o pré-processamento do texto
    processed_text = preprocess_text(text)
    return processed_text

def process_images(path: str) -> pd.DataFrame:
    """
    retorna DF pandas, com duas colunas 'Text' e 'Label';
    'Text': representa os textos extraido via OCR da imagem
    'Label' representa a classe/categoria/tipo/label original desta imagem
    """
    # Diretório com as imagens
    image_folder = os.listdir(path)
    # Lista para armazenar os dados antes de criar o DataFrame
    data = []
    # Itera sobre cada diretório de classe (label_dir) ['email', 'resume','scientific_publication']
    for label_dir in image_folder:
        # Caminho completo para o diretório atual de documentos
        label_path = os.path.join(path, label_dir)
        # Itera sobre os arquivos (imagem.png) dentro do diretório
        for file_name in os.listdir(label_path):
            processed_text = process_image(label_path, file_name)
            # Obtém o rótulo numérico correspondente ao tipo de documento (label)
            label = class_labels[label_dir]
            # Adiciona um dicionário à lista de dados
            data.append({'Text': processed_text, 'Label': label})
    # Cria o DataFrame final a partir da lista de dicionários
    df = pd.DataFrame(data)        
    return df

def process_dataset(path: str, parquet_file: str)-> pd.DataFrame:
    """
    carrega todo o dataset,
    pre processa o dataset de imagens extensão.png covertando para dataframe pandas,
    Dataframe já pronto para usar no treinamento do modelo
    """
    # Verifica se o arquivo Parquet já existe
    if os.path.exists(parquet_file):
        # Carrega os dados existentes se o arquivo Parquet já existir
        df = pd.read_parquet(parquet_file)
    else:
        df = process_images(path)
        # Salva o DataFrame em arquivo Parquet para uso futuro não precisar reprocessar e ja pegar pronto
        df.to_parquet(parquet_file, index=False)
    return df

def main():
    # Caminho para o diretório contendo as imagens a serem processadas
    path = "dataset/www-kaggle-com_datasets_ritvik1909_document-classification-dataset"
    # apos processar as imagens, Armazenar dados
    parquet_file = "df_document_texts.parquet"
    # Chamada da função para processar o dataset e obter o DataFrame resultante
    df = process_dataset(path, parquet_file)
    return df












