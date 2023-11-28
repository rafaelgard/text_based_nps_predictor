# Modelo de Previsão de NPS
[![Demonstração de Uso](src/images/streamlit_record.gif)](src/images/streamlit_record.gif)

Este projeto teve como objetivo criar um classificador utilizando técnicas de Processamento de Linguagem Natural (NLP) para realizar a previsão do Net Promoter Score (NPS) de um cliente com base em comentários coletados por meio de pesquisas de satisfação.

A partir de um comentário, é possível indicar a previsão da classificação do NPS do cliente, que pode ser: Detrator, Neutro ou Promotor.[1](https://pt.wikipedia.org/wiki/Net_Promoter_Score)

--------------------------------
## Aplicações:
Com a utilização do modelo, é possível agilizar a avaliação de comentários e a classificação do NPS, possibilitando uma resposta mais rápida aos clientes de uma empresa.

Se um cliente responde a uma pesquisa de satisfação e está muito insatisfeito, é possível detectar isso em tempo real com o modelo treinado e iniciar uma ação de prevenção, como o disparo de uma comunicação ou alerta na central de relacionamento com o cliente, por exemplo.

--------------------------------
## Modelos:
Foram treinados dois modelos com o intuito de permitir a comparação entre técnicas antigas e recentes relacionadas ao processamento de linguagem natural:

- Modelo 1: 
    - NLP: NLTK, Spacy, enelvo
    - Classificador: MultinomialNB

- Modelo 2: 
    - NLP: Transformers - 'neuralmind/bert-base-portuguese-cased'
    - Classificador: lightgbm

- O modelo 1, apesar de não ser um modelo tão recente é mais indicado a ser utilizado, pois devido ao desbalanceamento das classes ele possui um melhor desempenho nos testes F1 e um menor tempo de execução. Por padrão é utilizado o modelo 1.

--------------------------------
## Utilização?
É possível utilizar o modelo de 2 formas:

1: Diretamente no navegador com apenas um comando no terminal:

    "py -m streamlit run src/my_app_NPS.py"

2: Utilizando o método 'predict' da classe NLP para realizar a análise de um grande lote de comentários ao mesmo tempo.

--------------------------------
## Pacotes Requeridos
- python 3
- pandas
- numpy
- scipy
- sklearn
- transformers
- bert
- nltk
- spacy
- python -m spacy download pt_core_news_sm
- plotly
- wordcloud
- enelvo
