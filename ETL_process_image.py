import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2 as cv
import math
from scipy import ndimage
import pytesseract as pt
# importa arquivo interno
from ETL_process_text import main as preprocess_text

def correct_image_rotation(img: np.ndarray) -> np.ndarray:
    """
    Corrige a inclinação da imagem baseada nas linhas detectadas.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - Imagem corrigida.
    """
    img_edges = cv.Canny(img, 100, 100, apertureSize=3)
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        median_angle = np.median(angles)
        if median_angle != 0:
            img = ndimage.rotate(img, median_angle)
    return img

def extract_text_ocr(img: np.ndarray) -> pd.DataFrame:
    """
    Realiza OCR na imagem e retorna um DataFrame com os resultados.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - DataFrame com os resultados do OCR.
    """
    return pt.image_to_data(img, output_type='data.frame')

def filter_low_confidence_words(ocr_result: pd.DataFrame, confidence_threshold: int = 75) -> pd.DataFrame:
    """
    Filtra as palavras do OCR que têm uma confiança menor que o limiar especificado.
    Args:
    - ocr_result: DataFrame com os resultados do OCR.
    - confidence_threshold: Limiar de confiança para filtrar as palavras.
    Returns:
    - DataFrame filtrado com palavras de alta confiança.
    """
    return ocr_result[ocr_result.conf > confidence_threshold]

def calculate_mean_confidence(ocr_result: pd.DataFrame) -> float:
    """
    Calcula a média das confidências das palavras no resultado do OCR.
    Args:
    - ocr_result: DataFrame com os resultados do OCR.
    Returns:
    - Média das confidências.
    """
    return ocr_result["conf"].mean()

def apply_threshold_and_bilateral_filter(img: np.ndarray) -> np.ndarray:
    """
    Aplica binarização e filtro bilateral na imagem para tentar melhorar a qualidade do OCR.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - Imagem pré-processada com binarização e filtro bilateral aplicados.
    """
    ret, thresh1 = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
    img_blur = cv.bilateralFilter(thresh1, 9, 100, 100)
    return img_blur

def convert_ocr_result_to_text(ocr_result: pd.DataFrame) -> str:
    """
    Converte o resultado do OCR em uma string de texto.
    Args:
    - ocr_result: DataFrame com os resultados do OCR.
    Returns:
    - Texto extraído como uma string.
    """
    word_list = [str(word) for word in ocr_result['text']]
    filtered_word_list = [word for word in word_list if word != 'nan']
    text = ' '.join(filtered_word_list)
    return text

def preprocess_and_extract_text(img: np.ndarray) -> str:
    """
    Realiza OCR em uma imagem e retorna o texto extraído.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - Texto extraído da imagem.
    """
    # Extrai textos da imagem via OCR
    ocr_result = extract_text_ocr(img)
    # Filtra as palavras que têm uma confiança (conf) menor que 75
    ocr_result = filter_low_confidence_words(ocr_result)
    # Calcula a média das confidências (result_mean)
    result_mean = calculate_mean_confidence(ocr_result)
    # Aplica pré-processamento na imagem, para tentar melhorar a qualidade do OCR
    img_blur = apply_threshold_and_bilateral_filter(img)
    # Realiza OCR na imagem pré-processada
    result_after_preproc = extract_text_ocr(img_blur)
    # Filtra as palavras com confiança menor que 75
    result_after_preproc = filter_low_confidence_words(result_after_preproc)
    # Calcula a média das confidências (result_after_preproc_mean)
    result_after_preproc_mean = calculate_mean_confidence(result_after_preproc)
    # Compara a média das confidências antes e depois do pré-processamento
    if result_mean < result_after_preproc_mean:
        # entr as duas imagens, usa textos extraidos da imagem que tiver a melhor confiança 
        ocr_result = result_after_preproc
    # Converte o resultado do OCR em uma string de texto
    text = convert_ocr_result_to_text(ocr_result)
    return text

def process_image(image_path: str) -> tuple[str, str]:
    """
    Processa uma única imagem, realizando OCR, pré-processamento e extração de texto.
    Args:
    - image_path: Caminho completo para a imagem a ser processada.
    Returns:
    - Uma tupla contendo o texto extraído da imagem e classe da imagem.
    """
    image_file = Image.open(image_path)
    img = np.asarray(image_file)
    # Detectar e corrigir inclinação da imagem
    img = correct_image_rotation(img)
    # Extrai texto da imagem via OCR
    ocr_text = preprocess_and_extract_text(img)
    # Pré-processar o texto extraído
    preprocessed_text = preprocess_text(ocr_text)
    # Extrair o classe da imagem a partir do nome diretório que a imagem pertence
    class_img = os.path.basename(os.path.dirname(image_path))
    return preprocessed_text, class_img

def process_images(path_dataset: str, required_folders: list[str]) -> pd.DataFrame:
    """
    Processa imagens em um diretório, realizando OCR e pré-processamento.
    Args:
    - path_dataset: Caminho para o diretório contendo as imagens.
    Returns:
    - Um DataFrame pandas contendo o texto extraído de cada imagem e sua classe.
    """
    data = []
    for subdir, dirs, files in os.walk(path_dataset):
        name_subdir = subdir.rsplit('/', 1)[-1]
        if name_subdir in required_folders:
            print(f"processando arquivos do diretório {subdir} ... ")
            for file in files:
                img_text, class_img = process_image(os.path.join(subdir, file))
                # TODO: alterar nome 'doc_type' para class_img
                data.append({'text': img_text, 'doc_type': class_img})   
    df = pd.DataFrame(data)        
    return df

def process_dataset(path_dataset: str, path_df_parquet: str, required_folders: list[str])-> pd.DataFrame:
    """
    carrega todo o dataset,
    pre processa o dataset de imagens extensão.png covertando para dataframe pandas,
    Dataframe já pronto para usar no treinamento do modelo
    """
    # Verifica se o arquivo Parquet já existe
    if os.path.exists(path_df_parquet):
        # Carrega os dados existentes se o arquivo Parquet já existir
        df = pd.read_parquet(path_df_parquet)
    else:
        df = process_images(path_dataset, required_folders)
        # Salva o DataFrame em arquivo Parquet para uso futuro não precisar reprocessar e ja pegar pronto
        df.to_parquet(path_df_parquet, index=False)
    return df
