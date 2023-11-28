import pytest
from ..src import nlp_predictor
import spacy


@pytest.fixture
def nlp_object():
    '''Cria um objeto que será utilizado pelo pytest'''
    return nlp_predictor(model_type = 'BERT', retrain_model=False)


def test_lemmatize_words(nlp_object):
    '''Lemmatiza uma string e avalia se o retorno é o esperado'''
    input_text = "Será que vai funcionar?"
    expected_result = ['ser', 'que', 'ir', 'funcionar', '?']

    sp = spacy.load("pt_core_news_sm")
    result = nlp_object.lemmatize_words(input_text, sp)

    assert result == expected_result


def test_lemmatize_words_empty_string(nlp_object):
    '''Lemmatiza uma string vazia'''
    input_text = ""
    expected_result = []
    sp = spacy.load("pt_core_news_sm")
    result = nlp_object.lemmatize_words(input_text, sp)

    assert result == expected_result


def test_lemmatize_words_with_numbers(nlp_object):
    '''Lemmatiza texto contendo números e pontuação'''
    input_text = "Os preços são 10 reais cada."
    expected_result = ["o", "preço", "ser", "10", "real", "cada", "."]
    sp = spacy.load("pt_core_news_sm")

    result = nlp_object.lemmatize_words(input_text, sp)

    assert result == expected_result
