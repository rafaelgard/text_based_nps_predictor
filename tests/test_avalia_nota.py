import numpy as np
import pytest
from ..src import nlp_predictor


@pytest.fixture
def nlp_object():
    '''Cria um objeto que será utilizado pelo pytest'''
    return nlp_predictor(model_type = 'BERT', retrain_model=False)


def test_avalia_nota(nlp_object) -> None:
    '''Verifica se o método avalia nota está funcionando corretamente'''

    validacao = {'Promotor': np.random.randint(9, 10, size=30),
                 'Neutro': np.random.randint(7, 8, size=30),
                 'Detrator': np.random.randint(0, 6, size=30)
                 }

    assert np.all((nlp_object.avalia_nota(x) ==
                  'Promotor' for x in validacao['Promotor']))
    assert np.all((nlp_object.avalia_nota(x) ==
                  'Neutro' for x in validacao['Neutro']))
    assert np.all((nlp_object.avalia_nota(x) ==
                  'Detrator' for x in validacao['Detrator']))
