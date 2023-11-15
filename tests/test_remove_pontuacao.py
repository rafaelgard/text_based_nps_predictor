import numpy as np
import pytest
from ..src import nlp_predictor

@pytest.fixture
def nlp_object():
    return nlp_predictor(retrain_model=False)

def test_remove_pontuacao_com_pontuacao(nlp_object):
    input_review = "Isso é um teste! Com pontuação."
    expected_result = "Isso é um teste Com pontuação"

    result = nlp_object.remove_pontuacao(input_review)

    assert result == expected_result

def test_remove_pontuacao_sem_pontuacao(nlp_object):
    input_review = "Isso é um teste sem pontuação."
    expected_result = "Isso é um teste sem pontuação"

    result = nlp_object.remove_pontuacao(input_review)

    assert result == expected_result

def test_remove_pontuacao_com_numeros(nlp_object):
    input_review = "12345 Isso é um teste com números 67890."
    expected_result = " Isso é um teste com números "

    result = nlp_object.remove_pontuacao(input_review)

    assert result == expected_result

def test_remove_pontuacao_vazia(nlp_object):
    input_review = ""
    expected_result = ""

    result = nlp_object.remove_pontuacao(input_review)

    assert result == expected_result

def test_remove_pontuacao_com_objeto(nlp_object):
    input_review = {"key": "Isso é um teste com um objeto!"}
    expected_result = "key Isso é um teste com um objeto"

    result = nlp_object.remove_pontuacao(input_review)

    assert result == expected_result