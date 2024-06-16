# https://www.kaggle.com/datasets/shaz13/real-world-documents-collections

import os
from ETL_download_dataset_kaggle import download_dataset, process_zip_file
from ETL_text_cleaning import preprocess_text
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import math
from scipy import ndimage
import pytesseract as pt
import pandas as pd
from PIL import Image
import numpy as np
import cv2 as cv
import math
from scipy import ndimage
import pytesseract as pt
import pandas as pd
import os

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

def perform_ocr(img: np.ndarray) -> str:
    """
    Realiza OCR em uma imagem e retorna o texto extraído.
    - Leitura do texto original na imagem: Realiza OCR na imagem original.
    - Filtragem de palavras de baixa confiança: Remove palavras com confiança abaixo de um limiar (75).
    - Pré-processamento da imagem: Aplica binarização e filtro bilateral para melhorar a qualidade da imagem.
    - Leitura do texto na imagem pré-processada: Realiza OCR novamente na imagem pré-processada.
    - Comparação de resultados: Compara os resultados de OCR antes e depois do pré-processamento e escolhe o resultado com maior confiança média.
    Args:
    - img: Imagem representada como um array numpy.
    Returns:
    - Texto extraído da imagem.
    """
    ocr_result = pt.image_to_data(img, output_type='data.frame')
    # Filtra as palavras que têm uma confiança (conf) menor que 75 
    ocr_result = ocr_result[ocr_result.conf > 75]
    # Calcula a média das confidências (result_mean).
    result_mean = ocr_result["conf"].mean()
    # Aplica uma binarização na imagem (thresh1), onde todos os pixels com valor superior a 110 são convertidos para 255 (branco) e os demais para 0 (preto).
    ret,thresh1 = cv.threshold(img,110,255,cv.THRESH_BINARY)
    # Aplica um filtro bilateral na imagem binarizada para reduzir o ruído enquanto preserva as bordas.
    img_blur = cv.bilateralFilter(thresh1,9,100,100)
    # Realiza novamente o OCR na imagem, mas agora com ela já pré-processada (img_blur) 
    result_after_preproc = pt.image_to_data(img_blur, output_type='data.frame')
    # Filtra as palavras com confiança menor que 75. 
    result_after_preproc = result_after_preproc[result_after_preproc.conf > 75]
    # Calcula a média das confidências (result_after_preproc_mean).
    result_after_preproc_mean= result_after_preproc["conf"].mean()
    # Compara a média das confidências antes e depois do pré-processamento. 
    # Se a média das confidências após o pré-processamento (result_after_preproc_mean) for maior que a média antes do pré-processamento (result_mean), 
    if (result_mean<result_after_preproc_mean):
        # substitui o resultado original pelo resultado após o pré-processamento.
        ocr_result=result_after_preproc
    text = ' '.join([str(word) for word in ocr_result['text'] if str(word) != 'nan'])
    return text

def process_image(image_path: str) -> tuple[str, str]:
    """
    Processa uma única imagem, realizando OCR, pré-processamento e extração de texto.
    Args:
    - image_path: Caminho completo para a imagem a ser processada.
    Returns:
    - Uma tupla contendo o texto extraído da imagem e o tipo de documento.
    """
    image_file = Image.open(image_path)
    img = np.asarray(image_file)
    # Detectar e corrigir inclinação da imagem
    img = correct_image_rotation(img)
    # Realizar OCR na imagem
    ocr_text = perform_ocr(img)
    # Pré-processar o texto extraído
    preprocessed_text = preprocess_text(ocr_text)
    # Extrair tipo de documento do caminho da imagem
    doc_type = os.path.basename(os.path.dirname(image_path))
    return preprocessed_text, doc_type

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
        for file in files:
            if name_subdir in required_folders:
                img_text, doc_type = process_image(os.path.join(subdir, file))
                data.append({'text': img_text, 'doc_type': doc_type})   
    df = pd.DataFrame(data)        
    return df

def get_dataframe(path_dataset: str, path_df_parquet: str)-> pd.DataFrame:
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
        df = process_images(path_dataset)
        # Salva o DataFrame em arquivo Parquet para uso futuro não precisar reprocessar e ja pegar pronto
        df.to_parquet(path_df_parquet, index=False)
    return df

def main() -> pd.DataFrame:
    """
    Função principal que gerencia o fluxo de download e descompactação do dataset.
    """
    zip_file = 'real-world-documents-collections.zip'
    extract_to = 'dataset/real_world_documents_collections'
    link_dataset = 'shaz13/real-world-documents-collections'
    if not os.path.exists(zip_file):
        # Baixar o dataset
        download_dataset(link_dataset)
        # Verificar e processar o arquivo ZIP
        process_zip_file(zip_file, extract_to)
    path_dataset = "./dataset/real_world_documents_collections/docs-sm"
    path_df_parquet = './df_shaz13_real_world_documents_collections.parquet'
    df = get_dataframe(path_dataset, path_df_parquet)
    return df

# Executar a função principal
if __name__ == "__main__":
    df = main()
