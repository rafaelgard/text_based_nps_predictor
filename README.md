# Text Based Nps Predictor

[![CI](https://github.com/samuelcolvin/FastUI/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelgard/text_based_nps_predictor/actions?query=event%3Apush+branch%3Amain+workflow%3ACI)
[![versions](https://img.shields.io/pypi/pyversions/fastui.svg)](https://github.com/rafaelgard/text_based_nps_predictor)
[![license](https://img.shields.io/github/license/samuelcolvin/FastUI.svg)](https://github.com/rafaelgard/text_based_nps_predictor/blob/main/LICENSE)

![Alt text](src/images/NPS_wordcloud.png)
Este projeto teve como objetivo criar um classificador utlizando técnicas de NLP para fazer a previsão do NPS (Net Promoter Score) de um cliente baseado em comentários digitados pelo usuário.

A partir de um comentário é possível indicar a previsão da classificação do NPS do cliente que pode ser: Detrator, neutro ou promotor.[1](https://pt.wikipedia.org/wiki/Net_Promoter_Score)

## Exemplos:
É possível utilizar o modelo diretamente no navegador conforme o vídeo abaixo com apenas 1 comando no terminal:

"py -m streamlit run src/my_app_NPS.py"

[![Demonstração de Uso](src/images/streamlit_record.gif)](src/images/streamlit_record.gif)



Ou utilizar diretamente no código para avaliar diversos comentários ao mesmo tempo
![Alt text](src/images/example.png)

## Aplicações:
Com a utilização do modelo é possível agilizar a avaliação de comentários e a classificação do NPS de maneira a possibilitar uma resposta mais rápida aos clientes de uma empresa. 

Um cliente respondeu uma pesquisa de satisfação e está muito insatisfeito? É possível detectar isso em tempo real com o modelo treinado e iniciar uma ação de prevenção de cancelamento por exemplo o disparo de uma comunicação ou alerta na central de relacionamento com o cliente.

## Pacotes Requeridos
- python 3
- pandas
- numpy
- scipy
- sklearn
- nltk
- spacy
- python -m spacy download pt_core_news_sm
- plotly
- wordcloud
- enelvo
