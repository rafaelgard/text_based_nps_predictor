import numpy as np
from ..src import nlp_predictor

def test_carrega_classe() -> None:
    '''Verifica se conseguiu instanciar a classe corretamente'''
    nlp_object = nlp_predictor(retrain_model = False)
    assert nlp_object.retrain_model == False, 'Falhou ao instanciar a classe'

def test_carrega_modelo() -> None:
    nlp_object = nlp_predictor(retrain_model = False)
    assert nlp_object.retrain_model == False, 'Falhou ao carregar o modelo'

def test_treina_modelo() -> None:
    nlp_object = nlp_predictor(retrain_model = True)
    assert nlp_object.retrain_model == True, 'Falhou ao treinar o modelo'

def test_avalia_comentarios() -> None:
    comentarios_teste = ["Este serviço é excelente!",
                    "Estou insatisfeito com o atendimento e o produto é horrível",
                    "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                    ]

    gabarito = ['Promotor', 'Detrator', 'Detrator']

    nlp_object = nlp_predictor(retrain_model = False)
    results, _ = nlp_object.predict(comentarios_teste)

    for index, _ in enumerate(results):
        assert results[index] == gabarito[index], 'O modelo falhou ao avaliar comentários'
  
def test_tipo_retorno() -> None:
    comentarios_teste = ["Este serviço é excelente!",
                    "Estou insatisfeito com o atendimento e o produto é horrível",
                    "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                    ]

    nlp_object = nlp_predictor(retrain_model = False)
    results, _ = nlp_object.predict(comentarios_teste)
    assert isinstance(results, np.ndarray), 'o retorno do método predict não foi um array'
    