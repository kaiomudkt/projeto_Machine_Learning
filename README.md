# Como executar o ambiente

### usando docker:

1. limpando de container anteriores
```bash 
docker-compose down -v
```
2. subindo container 
```bash
docker-compose up --build
```
3. link jupyter:
- http://127.0.0.1:8888/tree?token=token_gerado
4. abrindo jupyter com Vs Code 
- precisa apenas escolher o kernel do jupyter e informar o token

# projeto_Machine_Learning

- LogisticRegression
- Random Forest Classifier
    - Random Forest é um ensemble de árvores de decisão que é frequentemente eficaz para classificação de textos.
- Support Vector Machine (SVM)
    - SVM é poderoso para classificação em espaços de alta dimensão, como texto.
- Naive Bayes
    - Especificamente, o Multinomial Naive Bayes é bem adaptado para classificação de texto.
- K-Nearest Neighbors (KNN)
    - KNN é um algoritmo de instância baseado em instância que pode ser útil, embora não seja tão eficiente para grandes conjuntos de dados de texto
- Gradient Boosting
    - Gradient Boosting é um algoritmo de boosting que pode fornecer alta precisão.

