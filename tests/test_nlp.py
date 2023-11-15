import numpy as np
import pytest
from ..src import nlp_predictor


@pytest.fixture
def nlp_object():
    '''Cria um objeto que será utilizado pelo pytest'''
    return nlp_predictor(retrain_model=False)


def test_carrega_classe(nlp_object) -> None:
    '''Verifica se conseguiu instanciar a classe corretamente'''
    assert nlp_object.retrain_model == False, 'Falhou ao instanciar a classe'


def test_modifica_atributo_classe(nlp_object) -> None:
    nlp_object.retrain_model = True
    assert nlp_object.retrain_model == True, 'Falhou ao treinar o modelo'


def test_avalia_comentarios(nlp_object) -> None:
    comentarios_teste = ["Este serviço é excelente!",
                         "Estou insatisfeito com o atendimento e o produto é horrível",
                         "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                         ]

    gabarito = ['Promotor', 'Detrator', 'Detrator']

    results, _ = nlp_object.predict(comentarios_teste)

    for index, _ in enumerate(results):
        assert results[index] == gabarito[index], 'O modelo falhou ao avaliar comentários'


def test_tipo_retorno(nlp_object) -> None:
    comentarios_teste = ["Este serviço é excelente!",
                         "Estou insatisfeito com o atendimento e o produto é horrível",
                         "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                         ]

    results, _ = nlp_object.predict(comentarios_teste)
    assert isinstance(
        results, np.ndarray), 'o retorno do método predict não foi um array'
