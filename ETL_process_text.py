# Apenas funções base de processamento de texto, com foco de gerar input para ML

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


#def process_text_from_image(label_path: str, file_name: str) -> str:
#    """
#    - carrega uma imagem.png
#    - extrai o texto da imagem via OCR
#    - realiza o pre processamento do texto
#    retorna o texto ja pre processado
#    """
#    # Carrega a imagem
#    image = Image.open(os.path.join(label_path, file_name))
#    # Usa Tesseract OCR para extrair o texto da imagem
#    text = pytesseract.image_to_string(image)
#    # Executa o pré-processamento do texto
#    processed_text = preprocess_text(text)
#    return processed_text

def main(text: str) -> str:
    """
    cleaning
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
