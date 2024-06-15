# https://www.kaggle.com/datasets/shaz13/real-world-documents-collections

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from ETL_download_dataset_kaggle import download_dataset, process_zip_file

def main() -> None:
    """
    Função principal que gerencia o fluxo de download e descompactação do dataset.
    """
    zip_file = 'real-world-documents-collections.zip'
    extract_to = 'dataset/real_world_documents_collections'
    link_dataset = 'shaz13/real-world-documents-collections'
    # Baixar o dataset
    download_dataset(link_dataset)
    # Verificar e processar o arquivo ZIP
    process_zip_file(zip_file, extract_to)

# Executar a função principal
if __name__ == "__main__":
    main()

