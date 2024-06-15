# https://www.kaggle.com/datasets/shaz13/real-world-documents-collections
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

def download_dataset() -> None:
    """
    Baixa o dataset do Kaggle após autenticação.
    Autentica na API do Kaggle usando as credenciais fornecidas no arquivo 
    kaggle.json e baixa o dataset especificado no diretório atual.
    Raises:
        FileNotFoundError: Se o arquivo kaggle.json não for encontrado.
    """
    # Configurar o caminho para o arquivo kaggle.json
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    # Verificar se o arquivo kaggle.json existe
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(f"O arquivo {kaggle_json_path} não foi encontrado. Certifique-se de que o arquivo kaggle.json está no diretório ~/.kaggle")
    # Configurar permissões para o arquivo kaggle.json
    os.chmod(kaggle_json_path, 0o600)
    # Inicializar a API do Kaggle
    api = KaggleApi()
    api.authenticate()
    dataset = 'shaz13/real-world-documents-collections'
    api.dataset_download_files(dataset, path='.', unzip=False)

def unzip_file(zip_file_path: str, extract_to: str) -> None:
    """
    Descompacta o arquivo ZIP em um diretório especificado.
    Args:
        zip_file_path (str): Caminho para o arquivo ZIP a ser descompactado.
        extract_to (str): Diretório onde os arquivos descompactados serão salvos.
    Raises:
        FileNotFoundError: Se o arquivo ZIP não for encontrado.
    """
    # Cria o diretório se ele não existir
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Arquivo {zip_file_path} descompactado com sucesso em {extract_to}')

def process_zip_file(zip_file: str, extract_to: str) -> None:
    """
    Verifica a existência de um arquivo ZIP e o descompacta.
    Args:
        zip_file (str): Caminho para o arquivo ZIP a ser verificado.
        extract_to (str): Diretório onde os arquivos descompactados serão salvos.
    """
    if os.path.exists(zip_file):
        unzip_file(zip_file, extract_to)
    else:
        print(f"O arquivo {zip_file} não foi encontrado.")

def main() -> None:
    """
    Função principal que gerencia o fluxo de download e descompactação do dataset.
    """
    zip_file = 'real-world-documents-collections.zip'
    extract_to = 'dataset/real_world_documents_collections'
    # Baixar o dataset
    download_dataset()
    # Verificar e processar o arquivo ZIP
    process_zip_file(zip_file, extract_to)

# Executar a função principal
if __name__ == "__main__":
    main()

