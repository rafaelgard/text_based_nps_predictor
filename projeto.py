'''Importação das bibliotecas'''

import pandas as pd
import numpy as np
import nltk
import spacy
import string
import plotly.express as px
from joblib import dump, load
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from enelvo.normaliser import Normaliser

def make_model():
    '''Importação dos dados'''
    df = pd.read_csv(r'database/dataframe_final.csv')

    # dropa os dados nulos
    df.dropna(inplace=True)

    # transforma todos os comentários para minúsculo
    df['Comentário'] = df['Comentário'].str.lower()

    # corrige a gramática
    df['Comentário'] = corrige_gramatica(df['Comentário'])

    # Remove a pontuação dos reviews
    df['Comentário']=df['Comentário'].apply(lambda x: remove_pontuacao(x))

    # lematiza a coluna comentário
    # Carregando o modelo em português do Brasil do spaCy
    nlp = spacy.load("pt_core_news_sm")
    
    df['Comentário'] = df['Comentário'].apply(lambda x: " ".join(lemmatize_words(x, nlp)))

    df.index=np.arange(df.shape[0])

    df['TARGET'] = df['Nota'].apply(lambda x: avalia_nota(x))

    df = df[df['Comentário'].isna()==False]

    df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

    # Dividindo os dados em conjunto de treinamento e conjunto de teste
    x = df['Comentário'].astype('str')
    y = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30, stratify=y)

    # instanciando e treinando o CountVectorizer
    stopWords = nltk.corpus.stopwords.words('portuguese')
    vectorizer = CountVectorizer(analyzer = "word", stop_words=stopWords) 
    X_train_vectorizer = vectorizer.fit_transform(X_train)
    X_test_vectorizer = vectorizer.transform(X_test)

    # instanciando o modelo
    model = MultinomialNB()

    # treinando o modelo nos dados de treinamento
    model.fit(X_train_vectorizer, y_train)

    # salvando o modelo
    dump((model, X_train_vectorizer, vectorizer), 'models/model.joblib')

    # fazendo as previsões no conjunto de teste
    y_pred = model.predict(X_test_vectorizer)

    # avaliando o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'\nAcurácia: {accuracy:.2f}')
    print(f'\nRelatório de Classificação:\n', report)

    cross_val_score = cross_val_score(estimator=model, X=X_test_vectorizer, y=y_test, cv=5, n_jobs=-1) #y_test, y_pred

    print('Resultado da validação cruzada')
    print(cross_val_score, '\n')

    print(f'Média do resultado da validação cruzada: {round(cross_val_score.mean(), 2)}')

    return model, vectorizer
    
def classifica_nps(model, vectorizer, comentario:str) -> np.array:
    "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

    if type(comentario) == str:
        comentario = np.array([comentario])

    comentarios_tfidf = vectorizer.transform(comentario)
    
    probabilities = model.predict_proba(comentarios_tfidf)
    resultado = model.predict(comentarios_tfidf)

    return resultado, probabilities


def lemmatize_words(text, nlp):
    '''Lematiza uma lista de palavras'''

    doc = nlp(text)
    
    lemmatized_words = [token.lemma_ for token in doc]
    
    return lemmatized_words

def corrige_gramatica(coluna):
    '''Conserta abreviações, potuações e remove emojis'''
    
    norm = Normaliser(tokenizer='readable', sanitize=True)
    
    # Usa a função vectorize do NumPy para aplicar a normalização a todos os elementos em uma única chamada
    normalizar = np.vectorize(norm.normalise)

    coluna_normalizada = normalizar(coluna)
    
    return coluna_normalizada

def remove_pontuacao(review):
    '''# Remove a pontuação dos comentários'''

    review = str(review)
    review = "".join([char for char in review if char not in string.punctuation])
    return review

def load_model():
    '''Carrega o modelo treinado'''
    model, X_train_vectorizer, vectorizer = load('models/model.joblib')
    return model, X_train_vectorizer, vectorizer

def avalia_nota(x):
    '''Avalia a nota e retorna a classificação do NPS'''

    if x >=9:
        classificacao = 'Promotor'
    
    elif x==7 or x==8:
        classificacao = 'Neutro'

    else:
        classificacao = 'Detrator'
    
    return classificacao

# Ative para fazer previsões em novos comentários
# new_comments = ["Este serviço é excelente!", 
#                 "Estou insatisfeito com o atendimento e o produto é horrível", 
#                 "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
#                 ]

# caso queira treinar o modelo
# model, vectorizer = make_model()

# caso queira importar o modelo já treinado
# model, X_train_vectorizer, vectorizer = load_model()

# resultado, probabilities = classifica_nps(model, vectorizer, new_comments)

# for index, comment in enumerate(new_comments):
#     print(f'Comentário: {new_comments[index]}')
#     print(f'Probabilidade Final: {resultado[index]}\n')
