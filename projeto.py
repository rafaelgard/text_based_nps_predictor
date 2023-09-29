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
    df = pd.read_csv('database/dataframe_final.csv')

    df.dropna(inplace=True)

    df['Comentário'] = df['Comentário'].str.lower()

    '''Analisando as palavras mais comuns'''

    # vectorizer = CountVectorizer(stop_words=stop_words)
    # bag_of_words = vectorizer.fit(df3['Comentário'].tolist())
    # bag_of_words = vectorizer.transform(df3['Comentário'].tolist())

    # print('Tamanho do vocabulário:{}' .format(len(vectorizer.vocabulary_)),'\n')
    # print('Tamanho do conteudo:\n{}' .format(vectorizer.vocabulary_))

    '''Removendo a pontuação'''
    # Esta função remove a pontuação da análise
    def remove_pontuacao(review):
        review = str(review)
        review = "".join([char for char in review if char not in string.punctuation])
        return review

    # Remove a pontuação dos reviews
    df['Comentário']=df['Comentário'].apply(lambda x: remove_pontuacao(x))

    # '''Lematização dos Reviews'''
    # # Esta função tokeniza a string contendo a análise
    # def lematiza(review):
    #     porter = PorterStemmer('portuguese')
    #     stemmed = [porter.stem(word) for word in review]
    #     return stemmed
    

    # breakpoint()
    # df['Comentário']=df['Comentário'].apply(lambda x: lematiza(x))

    # Inicialize o WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Função para lematizar uma lista de palavras
    def lematizar_palavras(lista_de_palavras):
        review = [lemmatizer.lemmatize(palavra) for palavra in lista_de_palavras]
        # review = " ".join([char for char in review ])
        review = " ".join([char for char in review if char not in ['gbarbosa', 'gb', 'bretas', 'cencosud', 'gbaborsa', 'gbarboza', 'g barbosa', 'g barboza']])
        return review

    # Aplique a lematização à coluna 'Comentário' em seu DataFrame
    df['Comentário'] = df['Comentário'].apply(lambda x: lematizar_palavras(nltk.word_tokenize(x, language='portuguese')))


    df.index=np.arange(len(df))
    print('df.head()')
    df.head()

    def avalia_nota(x):
        if x >=9:
            return 'Promotor'
        
        elif x==7 or x==8:
            return 'Neutro'

        else:
            return 'Detrator'

    df['TARGET']=df['Nota'].apply(lambda x: avalia_nota(x))

    df.drop(columns=['Nota'], inplace=True)

    # df = df[['Comentário', 'Categoria', 'Status', 'TARGET']]
    df = df[df['Comentário'].isna()==False]

    df.drop_duplicates(keep='last', inplace=True, ignore_index=True)
    # exit()

    # breakpoint()

    # Divida os dados em recursos (X) e a variável de destino (y)
    x = df['Comentário'].astype('str')
    y = df['TARGET']

    print('y.value_counts()')
    print(y.value_counts())

    # breakpoint()
    # Divida os dados em conjunto de treinamento e conjunto de teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(x_smoteenn, y_smoteenn, test_size=0.2, random_state=30)

    # Use TfidfVectorizer ou CountVectorizer para converter os comentários de texto em recursos numéricos
    # vectorizer = TfidfVectorizer(stop_words=stopWords)
    stopWords = get_stop_words('portuguese')
    vectorizer = CountVectorizer(analyzer = "word", stop_words=stopWords) 
    X_train_vectorizer = vectorizer.fit_transform(X_train)
    X_test_vectorizer = vectorizer.transform(X_test)

    model = MultinomialNB()
    # model = XGBClassifier(random_state=30, seed=30, n_jobs=-1)

    # Treine o modelo nos dados de treinamento
    model.fit(X_train_vectorizer, y_train)

    # salva o modelo
    dump((model, X_train_vectorizer, vectorizer), 'models/model.joblib')

    # Faça previsões no conjunto de teste
    y_pred = model.predict(X_test_vectorizer)

    # Avalie o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Acurácia: {accuracy:.2f}')
    print('Relatório de Classificação:\n', report)

    return model, vectorizer
    

# new_comments_tfidf = tfidf_vectorizer.transform(new_comments)
# probabilities = model.predict_proba(new_comments_tfidf)
# print('Probabilidades de cancelamento:\n', probabilities)
# print(f'Probabilidade Final: {model.predict(new_comments_tfidf)}')

def classifica_nps(model, vectorizer, comentario:str) -> np.array:
    "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

    if type(comentario) == str:
        comentario = np.array([comentario])

    comentarios_tfidf = vectorizer.transform(comentario)
    
    probabilities = model.predict_proba(comentarios_tfidf)
    resultado = model.predict(comentarios_tfidf)
    # print('type(resultado)')
    # print(type(resultado))
    # print('probabilities')
    # print(probabilities)
    # print('probabilities 0')
    # print(probabilities[0][0])
    
    # exit()
    return resultado, probabilities

def load_model():
    '''Carrega o modelo treinado'''
    model, X_train_vectorizer, vectorizer = load('models/model.joblib')
    return model, X_train_vectorizer, vectorizer

# # Para fazer previsões em novos comentários
new_comments = ["Este serviço é excelente!", 
                "Estou insatisfeito com o atendimento e o produto é horrível", 
                "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                ]


# model, vectorizer = make_model()
# model, X_train_vectorizer, vectorizer = load_model()

# resultado, probabilities = classifica_nps(model, vectorizer, new_comments)

# for index, comment in enumerate(new_comments):
#     print(f'Comentário: {new_comments[index]}')
#     print(f'Probabilidade Final: {resultado[index]}\n')
# breakpoint()