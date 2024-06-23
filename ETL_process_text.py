# Apenas funções base de processamento de texto, com foco de gerar input para ML
"""
Conversão para Minúsculas 

Substituição de Quebras de Linha e Tabulações 

Remoção de Espaços Extras 

Remoção de Números

Remoção de Pontuação 

Tokenização (word_tokenize(text)):
 - word_tokenize(text) utiliza a função word_tokenize do NLTK para dividir o texto em tokens individuais (palavras e pontuações). Isso é fundamental para trabalhar com o texto em nível de palavra.
Remoção de Pontuação Adicional e Stopwords

Lemmatização:
- A lematização é realizada usando o WordNetLemmatizer do NLTK. Cada palavra restante após a remoção de stopwords é lematizada, ou seja, reduzida à sua forma base (lema). Isso ajuda na normalização do texto, reduzindo variações morfológicas e simplificando o vocabulário.

Concatenação dos Tokens:
- todos os tokens lematizados são unidos em uma única string, onde cada token é separado por um espaço em branco. Esta string processada é então retornada como a saída da função.
"""
import os
from typing import List
from string import punctuation
from nltk.stem import WordNetLemmatizer 
import re
import string

# NLTK (Natural Language Toolkit) 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import pytesseract
# Download dos recursos necessários do NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# conjunto de palavras stopwords para o idioma inglês 
# Lista de palavras que são consideradas stopwords (palavras vazias) para o idioma inglês. 
stopwords_list: List[str] = stopwords.words("english") 

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
    - utiliza a função word_tokenize do NLTK para dividir o texto em tokens individuais (palavras e pontuações). 
    - tem como objetivo processar uma sentença de texto, aplicando algumas regras específicas para transformar e limpar os tokens resultantes.
    Isso é fundamental para trabalhar com o texto em nível de palavra.
    Args:
        text (str): O texto a ser tokenizado.
    Returns:
        List[str]: Lista de tokens gerada a partir do texto.
    """
    tokens = word_tokenize(text)
    return tokens

def remove_punctuation(tokens: List[str]) -> List[str]:
    """
    Remove pontuações da lista de tokens.
    Args:
        tokens (List[str]): Lista de tokens a serem processados.
    Returns:
        List[str]: Lista de tokens sem pontuações.
    """
    # Lista de pontuações do módulo string
    punctuations = string.punctuation
    # Filtrar tokens removendo pontuações
    tokens = [token for token in tokens if token not in punctuations]
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords da lista de tokens.
    Args:
        tokens (List[str]): Lista de tokens a serem processados.
        stopwords_list (List[str]): Lista de stopwords a serem removidas.
    Returns:
        List[str]: Lista de tokens sem stopwords.
    """
    tokens = [token for token in tokens if token.lower() not in stopwords_list]
    return tokens

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Realiza a lematização dos tokens usando WordNetLemmatizer do NLTK.
    - A lematização é realizada usando o WordNetLemmatizer do NLTK. 
    - Cada palavra restante após a remoção de stopwords é lematizada, ou seja, reduzida à sua forma base (lema). 
    - Isso ajuda na normalização do texto, reduzindo variações morfológicas e simplificando o vocabulário.
    Args:
        tokens (List[str]): Lista de tokens a serem lematizados.
    Returns:
        List[str]: Lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def main(text: str) -> str:
    """
    - cleaning
    - Realiza o pré-processamento completo do texto, incluindo minúsculas,
    - remoção de caracteres especiais, tokenização, remoção de pontuações
    - e stopwords, e lematização.
    - todos os tokens lematizados são unidos em uma única string, 
    - onde cada token é separado por um espaço em branco. 
    - Esta string processada é então retornada como a saída da função.
    Args:
        text (str): O texto original a ser pré-processado.
    Returns:
        str: O texto pré-processado final, pronto para uso em análises adicionais.
    """
    # Aplica as funções em sequência
    text = lowercase_and_remove_special_characters(text)
    tokens = tokenize_text(text)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    final_text = lemmatize_tokens(tokens)
    return " ".join(final_text)
