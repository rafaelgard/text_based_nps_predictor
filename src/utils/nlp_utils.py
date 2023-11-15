from joblib import load
import numpy as np


def classifica_nps(model, vectorizer, comentario: str) -> np.array:
    "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

    if isinstance(comentario, str):
        comentario = np.array([comentario])

    comentarios_tfidf = vectorizer.transform(comentario)

    probabilities = model.predict_proba(comentarios_tfidf)
    resultado = model.predict(comentarios_tfidf)

    return resultado, probabilities


def load_model():
    '''Carrega o modelo treinado'''

    model, X_train_vectorizer, vectorizer = load('models/model.joblib')

    return model, X_train_vectorizer, vectorizer
