# Como executar o ambiente

### Usando Docker:

1. Criar uma conta no Kaggle para fazer download do dataset e configurar suas chaves de API do Kaggle (CREATE NEW API TOKEN):
    1.1. Acesse: [Configurações da Conta Kaggle](https://www.kaggle.com/settings/account)
    1.2. Clique no botão "Create New Token" para baixar o arquivo `kaggle.json`.
    1.3. Dentro deste repositório coloque em: `.kaggle/kaggle.json`

2. Configurar permissões (apenas para Linux/macOS):
    ```bash
    chmod 600 .kaggle/kaggle.json
    ```

3. Instalar a biblioteca Kaggle:
    ```bash
    pip install kaggle
    ```

4. Limpar containers anteriores:
    ```bash
    docker-compose down -v
    ```

5. Subir container:
    ```bash
    docker-compose up --build
    ```

6. Acessar o Jupyter:
    - Abra o link gerado no terminal, similar a: [http://127.0.0.1:8888/tree?token=token_gerado](http://127.0.0.1:8888/tree?token=token_gerado)

7. Abrir Jupyter com VS Code:
    - Escolha o kernel do Jupyter e informe o token gerado.

    - Link do Jupyter: [http://127.0.0.1:8888/](http://127.0.0.1:8888/)
    - Token é a senha fornecida no terminal.

8. Após, abra seu navegador em: [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab)

# Projeto de Machine Learning

- **Logistic Regression**
- **Random Forest Classifier**
    - Random Forest é um ensemble de árvores de decisão que é frequentemente eficaz para classificação de textos.
- **Support Vector Machine (SVM)**
    - SVM é poderoso para classificação em espaços de alta dimensão, como texto.
- **Naive Bayes**
    - Especificamente, o Multinomial Naive Bayes é bem adaptado para classificação de texto.
- **K-Nearest Neighbors (KNN)**
    - KNN é um algoritmo baseado em instância que pode ser útil, embora não seja tão eficiente para grandes conjuntos de dados de texto.
- **Gradient Boosting**
    - Gradient Boosting é um algoritmo de boosting que pode fornecer alta precisão.
