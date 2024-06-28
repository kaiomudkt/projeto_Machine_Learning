# este arquivo faz download de dataset a partir do Kaggle

# https://www.kaggle.com/datasets
# Faça login na sua conta Kaggle.
# Clique na sua foto de perfil no canto superior direito e selecione "My Account".
# Role para baixo até a seção "API" e clique no botão "Create New API Token". Isso fará o download de um arquivo chamado kaggle.json.
# Configurar permissões (apenas para Linux/macOS):
# Certifique-se de que o arquivo kaggle.json tenha as permissões corretas para que apenas o seu usuário possa lê-lo:
# chmod 600 ~/.kaggle/kaggle.json
# Instale a biblioteca kaggle:
# pip install kaggle

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(link_dataset: str, destination_path: str='.') -> None:
    """
    Baixa o dataset do Kaggle após autenticação.
    Autentica na API do Kaggle usando as credenciais fornecidas no arquivo 
    kaggle.json e baixa o dataset especificado no diretório atual.
    Raises:
        FileNotFoundError: Se o arquivo kaggle.json não for encontrado.
    """
    try:
        # Configurar o caminho para o arquivo kaggle.json
        kaggle_json_path = os.path.expanduser('./.kaggle/kaggle.json')
        # Verificar se o arquivo kaggle.json existe
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"O arquivo {kaggle_json_path} não foi encontrado. Certifique-se de que o arquivo kaggle.json está no diretório ~/.kaggle")
        # Configurar permissões para o arquivo kaggle.json
        os.chmod(kaggle_json_path, 0o600)
        # Inicializar a API do Kaggle
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(link_dataset, path=destination_path, unzip=False)
    except Exception as e:
        print(f'Erro ao realizar download: {e}')

def unzip_file(zip_file_path: str, extract_to: str) -> None:
    """
    Descompacta o arquivo ZIP em um diretório especificado.
    Args:
        zip_file_path (str): Caminho para o arquivo ZIP a ser descompactado.
        extract_to (str): Diretório onde os arquivos descompactados serão salvos.
    Raises:
        FileNotFoundError: Se o arquivo ZIP não for encontrado.
    """
    if os.path.exists(zip_file_path):
        # Se o diretório de destinho se ele não existir, então será criado
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f'Arquivo {zip_file_path} descompactado com sucesso em {extract_to}')
    else:
        print(f"O arquivo {zip_file_path} não foi encontrado.")

def get_dataset(zip_file_name: str, link_kaggle_dataset: str, path_zip: str, path_dataset: str):
    if not os.path.exists(zip_file_name):
        download_dataset(link_kaggle_dataset, path_zip)
    if os.path.exists(zip_file_name):
        unzip_file(zip_file_name, path_dataset)
        
def main() -> None:
    """
    Função principal que gerencia o fluxo de download e descompactação do dataset.
    """
    path_dataset: str = "dataset/www-kaggle-com_datasets_ritvik1909_document-classification-dataset"
    path_df_parquet: str = './DF_process_dataset_ritvik1909_V2.parquet'
    path_zip: str = "./dataset/compactados"
    zip_file: str = f"{path_zip}/document-classification-dataset.zip"
    
    link_kaggle_dataset: str = 'ritvik1909/document-classification-dataset' 
    destination_path: str = "./dataset/compactados"
    # Baixar o dataset
    download_dataset(link_kaggle_dataset, destination_path)
    # Descompacta o arquivo ZIP
    unzip_file(zip_file, path_dataset)

# Executar a função principal
if __name__ == "__main__":
    main()

