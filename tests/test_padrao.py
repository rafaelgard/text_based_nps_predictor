import pandas as pd
import numpy as np
from joblib import dump, load
# from utils.nlp_utils import classifica_nps, load_model
# from ..utils.nlp_utils import * 
# from src.projeto.nlp import nlp_predictor
# from ..src.projeto.nlp import *
import sys
import os
# sys.path.append(f'D:\Projetos\text_based_nps_predictor')
for x in sys.path:
    print(x) 
# from ..text_based_nps_predictor. import *
# from TEXT_BASED_NPS_PREDICTOR.src.projeto.nlp import *

# Adicione o diretório src ao sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))



# def test_sondas() -> None:
#     '''Verifica se está encontrando a solução inicial no final da construtiva 
#     em todas as instâncias'''

#         assert np.all(atividades == np.arange(1, n+1)) == True
#         for indice_atividade in self_analise_atividades.keys():

#             # testes de tipo de valores
#             assert isinstance(self_analise_atividades[indice_atividade]['instante_de_termino'], float)
#             assert isinstance(self_analise_atividades[indice_atividade]['tempo_de_processamento'], float)
#             assert isinstance(self_analise_atividades[indice_atividade]['sonda_associada'], int)
#             assert isinstance(self_analise_atividades[indice_atividade]['alocada'], bool)

#             assert self_analise_atividades[indice_atividade]['instante_de_termino'] > 0
#             assert self_analise_atividades[indice_atividade]['tempo_de_processamento'] > 0
#             assert self_analise_atividades[indice_atividade]['sonda_associada'] > 0
#             assert self_analise_atividades[indice_atividade]['alocada'] == True


# model, X_train_vectorizer, vectorizer = load_model()

# def test_prever_categoria_comentario_positivo():
#     comentario = "Este é um ótimo produto!"
    
#     resultado, probabilities  = classifica_nps(model, vectorizer, comentario)
   
#     assert resultado == "Promotor"

# def test_prever_categoria_comentario_negativo():
#     comentario = "Este produto é péssimo."

#     resultado, probabilities  = classifica_nps(model, vectorizer, comentario)
   
#     assert resultado == "Detrator"

# # novos comentários que serão utilizados para testar o modelo gerado
# novos_comentários = ["Este serviço é excelente!",
#                      "Estou insatisfeito com o atendimento e o produto é horrível",
#                      "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
#                      ]

# model, X_train_vectorizer, vectorizer = load_model()

# resultado, probabilities = classifica_nps(model, vectorizer, novos_comentários)

# for index, comment in enumerate(novos_comentários):
#     print(f'Comentário: {novos_comentários[index]}')
#     print(f'Resultado previsto: {resultado[index]}\n')

