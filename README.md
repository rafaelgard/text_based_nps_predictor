# text_based_nps_predictor

O objetivo desde projeto é analisar e criar um classificador para avaliar comentários e fazer a previsão da classificação do NPS baseado no texto presente no comentário.

## Foi treinado e avaliado o seguinte classificador:
- MultinomialNB

## Pacotes Requeridos
- python 3
- pandas
- numpy
- scipy
- sklearn
- nltk
- spacy
- plotly
- wordcloud
- enelvo

## Exemplo de utilização do modelo treinado:

Comentário: Este serviço é excelente!
Resultado previsto: Promotor

Comentário: Estou insatisfeito com o atendimento e o produto é horrível
Resultado previsto: Detrator

Comentário: Péssimo cartão! Não fui bem atendida e meu limite é baixo
Probabilidade Final: Detrator