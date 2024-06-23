# OBJETIVOS DESTE ARQUIVO
# processa imagens (png, jpg)
# extrai textos das imagem via OCR
# processa textos
# gera dataframe para ser usado em uma ML

# o processamento considera que seu dataset esta no seguinte formado
# diretório raiz do dataset, e cada sub diretório, é uma classe de imagem, com suas respectivas 'n' quantidade de imagens
# a classe significa o tipo da imagem, ou seja, target
import os
import pandas as pd
import numpy as np
import cv2 as cv
import math
import pickle
import pytesseract
from scipy import ndimage
from PIL import Image, ImageDraw
# importa arquivo interno
from ETL_process_text import main as preprocess_text
from typing import List, Tuple, Union, Dict

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

def extract_text_ocr(img: np.ndarray) -> dict:
    """
    Realiza OCR na imagem e retorna um DataFrame com os resultados.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - DataFrame com os resultados do OCR.
    """
    ocr_dict = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    return ocr_dict

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
    # TODO: analisar word_list, e verificar se tem string vazias ou 'nan' para remover sua linha toda, para nao ficar seu eixo x e y
    # TODO: talvez essa condição if word != 'nan', esteja errada e deixando passar nan que nao deveria
    filtered_word_list = [word for word in word_list if word != 'nan']
    text = ' '.join(filtered_word_list)
    return text

def pre_process_image(ocr_dict: dict)  -> Tuple[pd.DataFrame, float]:
    # Converter o dicionário em um DataFrame
    df_ocr_result = pd.DataFrame(ocr_dict)
    # Remove linhas com valores NaN e redefine os índices
    df_ocr_result = df_ocr_result.dropna().reset_index(drop=True)
    # Seleciona colunas com valores float
    float_cols = df_ocr_result.select_dtypes('float').columns
    # Arredonda os valores das colunas float para zero casas decimais e os converte para inteiros
    df_ocr_result[float_cols] = df_ocr_result[float_cols].round(0).astype(int)
    # Substitui strings vazias ou espaços em branco por NaN
    df_ocr_result = df_ocr_result.replace(r'^\s*$', np.nan, regex=True)
    # Remove linhas com valores NaN e redefine os índices
    # TODO: talvez tenha que drop todas as linhas onde a coluna text for vazia
    df_ocr_result = df_ocr_result.dropna().reset_index(drop=True)
    # Filtra as palavras que têm uma confiança (conf) menor que 75
    df_ocr_result = filter_low_confidence_words(df_ocr_result)
    # Calcula a média das confidências (result_mean)
    result_mean = calculate_mean_confidence(df_ocr_result)
    return df_ocr_result, result_mean

def preprocess_and_extract_text(img: np.ndarray) -> Tuple[str, Dict]:
    """
    Realiza OCR em uma imagem e retorna o texto extraído.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - Texto extraído da imagem.
    """
    # Extrai textos da imagem via OCR
    ocr_dict = extract_text_ocr(img)
    # Pre processamento da imagem
    df_ocr_result, result_mean = pre_process_image(ocr_dict)
    # Agora vamos usar a mesma imagem, mas com um tratamento antes, para verificar se melhora o OCR
    # Aplica pré-processamento na imagem, para tentar melhorar a qualidade do OCR
    img_blur = apply_threshold_and_bilateral_filter(img)
    # Realiza novamente OCR na imagem pré-processada (img_blur)
    dict_result_after_preproc = extract_text_ocr(img_blur)
    # Pre processamento da imagem
    df_ocr_after_preproc, result_after_preproc_mean = pre_process_image(df_ocr_result)
    # Compara a média das confidências antes e depois do pré-processamento
    if result_mean < result_after_preproc_mean:
        # entr as duas imagens, usa textos extraidos da imagem que tiver a melhor confiança 
        df_ocr_result = df_ocr_after_preproc
        ocr_dict = dict_result_after_preproc
    # Converte o resultado do OCR em uma string de texto
    text = convert_ocr_result_to_text(df_ocr_result)
    return text, ocr_dict

def process_image(image_path: str, make_correct_img_rotation: bool=True) -> tuple[str, str, Dict]:
    """
    Processa uma única imagem, realizando OCR, pré-processamento e extração de texto.
    Args:
    - image_path: Caminho completo para a imagem a ser processada.
    Returns:
    - Uma tupla contendo o texto extraído da imagem e classe da imagem, e dict_ocr_result.
    """
    image_file = Image.open(image_path)
    img = np.asarray(image_file)
    if make_correct_img_rotation:
        # Detectar e corrigir inclinação da imagem
        img = correct_image_rotation(img)
    # Extrai texto da imagem via OCR
    text, dict_ocr_result_teste = preprocess_and_extract_text(img)
    # Pré-processar o texto extraído
    preprocessed_text = preprocess_text(text)
    # Extrair o classe da imagem a partir do nome diretório que a imagem pertence
    class_img = os.path.basename(os.path.dirname(image_path))
    return preprocessed_text, class_img, dict_ocr_result_teste

def serialize_df(df):
    return pickle.dumps(df)

def process_images(path_dataset: str, required_folders: dict) -> pd.DataFrame:
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
                img_text, class_img, dict_ocr_result = process_image(os.path.join(subdir, file))
                data.append({'text': img_text, 'class_img': class_img, 'name': file, 'dict_ocr': dict_ocr_result, 'class_number': required_folders[name_subdir]})   
    df = pd.DataFrame(data)        
    return df

def draw_bounding_boxes(image: Image, ocr_df: pd.DataFrame) -> Image:
    """
    Desenha caixas delimitadoras em uma imagem com base nos resultados do OCR.
    Args:
    - image: Imagem PIL onde as caixas delimitadoras serão desenhadas.
    - ocr_df: df bruto extraido da imagem via OCR
    - coordinates: DataFrame do pandas contendo as colunas 'left', 'top', 'width', 'height'.
    Returns:
    - Imagem PIL com as caixas delimitadoras desenhadas.
    """
    # Extrai as coordenadas das caixas delimitadoras
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes: List[Tuple[int, int, int, int]] = []
    # Converte as coordenadas em caixas delimitadoras reais
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # A linha vem no formato (left, top, width, height)
        actual_box = (x, y, x+w, y+h)  # Transformamos para (left, top, left+width, top+height) para obter a caixa real
        actual_boxes.append(actual_box)
    # Verifica o modo da imagem e converte para 'RGB' se necessário
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Desenha as caixas delimitadoras na imagem
    draw = ImageDraw.Draw(image, "RGB")
    for box in actual_boxes:
        draw.rectangle(box, outline='red')
    return image

def process_dataset(path_dataset: str, path_df_parquet: str, required_folders: dict)-> pd.DataFrame:
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
