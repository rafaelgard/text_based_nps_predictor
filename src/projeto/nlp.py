'''Importação das bibliotecas'''

import string
from joblib import dump, load
import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from enelvo.normaliser import Normaliser


class nlp_predictor:
    def __init__(self, retrain_model):
        '''Classe que permite a utilização dos métodos implementados'''
        # se retrain_model=True, o modelo é retreinado,
        # caso contrário é carregado um modelo já treinado
        self.retrain_model = retrain_model

    def make_model(self):
        '''Importação dos dados'''
        df = pd.read_csv(r'src/database/dataframe_final.csv')

        # dropa os dados nulos
        df.dropna(inplace=True)

        # transforma todos os comentários para minúsculo
        df['Comentário'] = df['Comentário'].str.lower()

        # corrige a gramática
        df['Comentário'] = self.corrige_gramatica(df['Comentário'])

        # Remove a pontuação dos reviews
        df['Comentário'] = df['Comentário'].apply(self.remove_pontuacao)

        # lematiza a coluna comentário
        # Carregando o modelo em português do Brasil do spaCy
        nlp = spacy.load("pt_core_news_sm")

        df['Comentário'] = df['Comentário'].apply(
            lambda x: " ".join(self.lemmatize_words(x, nlp)))

        df.index = np.arange(df.shape[0])

        df['TARGET'] = df['Nota'].apply(lambda x: self.avalia_nota(x))

        df = df[df['Comentário'].isna() == False]

        df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

        # Dividindo os dados em conjunto de treinamento e conjunto de teste
        x = df['Comentário'].astype('str')
        y = df['TARGET']

        # separando os dados em conjunto de teste e treino
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=30, stratify=y)

        # instanciando e treinando o CountVectorizer
        stopWords = nltk.corpus.stopwords.words('portuguese')
        vectorizer = CountVectorizer(analyzer="word", stop_words=stopWords)
        X_train_vectorizer = vectorizer.fit_transform(X_train)
        X_test_vectorizer = vectorizer.transform(X_test)

        # instanciando o modelo
        model = MultinomialNB()

        # treinando o modelo nos dados de treinamento
        model.fit(X_train_vectorizer, y_train)

        # salvando o modelo
        dump((model, X_train_vectorizer, vectorizer), 'src/models/model.joblib')

        # fazendo as previsões no conjunto de teste
        y_pred = model.predict(X_test_vectorizer)

        # avaliando o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'\nAcurácia: {accuracy:.2f}')
        print(f'\nRelatório de Classificação:\n', report)

        cross_val_score_result = cross_val_score(
            estimator=model, X=X_test_vectorizer, y=y_test, cv=5, n_jobs=-1)

        print('Resultado da validação cruzada')
        print(cross_val_score_result, '\n')

        print(
            f'Média do resultado da validação cruzada: {round(cross_val_score_result.mean(), 2)}\n')

        return model, vectorizer

    def create_wordcloud(self, column):
        '''Cria uma word cloud de uma coluna específica do dataframe'''

        '''Importação dos dados'''
        df = pd.read_csv(r'src/database/dataframe_final.csv', usecols=column)

        # dropa os dados nulos
        df.dropna(inplace=True)

        # transforma todos os comentários para minúsculo
        df[column] = df[column].str.lower()

        # corrige a gramática
        df[column] = self.corrige_gramatica(df[column])

        # Remove a pontuação dos reviews
        df[column] = df[column].apply(self.remove_pontuacao)
        
        # lematiza a coluna comentário
        # Carregando o modelo em português do Brasil do spaCy
        nlp = spacy.load("pt_core_news_sm")

        df[column] = df[column].apply(
            lambda x: " ".join(self.lemmatize_words(x, nlp)))

        df.index = np.arange(df.shape[0])

        df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

        # juntando todos os comentários para construir a wordcloud
        todos_os_comentarios = "".join(comentario for comentario in df[column])

        # instanciando a wordcloud
        wordcloud = WordCloud(stopwords=nltk.corpus.stopwords.words('portuguese'),
                              background_color='black', width=1600,
                              height=800, max_words=1000,  max_font_size=500,
                              min_font_size=1).generate(todos_os_comentarios)  # mask=mask,

        # exibindo a wordcloud
        _, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        plt.imshow(wordcloud)
        plt.savefig('images/new_wordcloud.png')

    def classifica_nps(self, model, vectorizer, data) -> (np.array, np.array):
        "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

        comentarios_tfidf = vectorizer.transform(data)

        probabilities = model.predict_proba(comentarios_tfidf)
        resultado = model.predict(comentarios_tfidf)

        return resultado, probabilities

    def lemmatize_words(self, text, nlp):
        '''Lematiza uma lista de palavras'''

        doc = nlp(text)

        lemmatized_words = [token.lemma_ for token in doc]

        return lemmatized_words

    def corrige_gramatica(self, coluna):
        '''Conserta abreviações, potuações e remove emojis'''

        norm = Normaliser(tokenizer='readable', sanitize=True)

        # Usa a função vectorize do NumPy para aplicar a normalização
        # a todos os elementos em uma única chamada
        normalizar = np.vectorize(norm.normalise)

        coluna_normalizada = normalizar(coluna)

        return coluna_normalizada

    def remove_pontuacao(self, review):
        '''# Remove a pontuação dos comentários'''

        review = str(review)
        review = "".join(
            [char for char in review if char not in string.punctuation
             and char not in string.digits])
        return review

    def load_model(self):
        '''Carrega o modelo treinado'''
        model, X_train_vectorizer, vectorizer = load(
            r'src\models\model.joblib')
        return model, X_train_vectorizer, vectorizer

    def avalia_nota(self, x):
        '''Avalia a nota e retorna a classificação do NPS'''

        if x >= 9:
            classificacao = 'Promotor'

        elif x in (7, 8):
            classificacao = 'Neutro'

        else:
            classificacao = 'Detrator'

        return classificacao

    def get_model(self):
        '''Carrega o modelo e o vectorizer levando em consideração a escolha do usuário de
        treinar o modelo ou utilizar um modelo carregado'''

        if self.retrain_model:
            model, vectorizer = self.make_model()

        else:
            model, _, vectorizer = self.load_model()

        return model, vectorizer

    def predict(self, data):
        '''Avalia uma lista ou array de strings e faz a 
        previsão da classificação'''

        comentario_valido = True

        # identifica se a variável data é lista, array ou string
        types_tests = (isinstance(data, list), isinstance(
            data, np.ndarray), isinstance(data, str))

        if not any(types_tests):
            print('O comentário inválido! Verifique a documentação do projeto!')
            comentario_valido = False

        if comentario_valido:
            if isinstance(data, str):
                comentario = np.array([data])
                comentario_valido = True
    
            else:
                comentario = data

            model, vectorizer = self.get_model()
            resultado, probabilities = self.classifica_nps(
                model, vectorizer, comentario)

            return resultado, probabilities
        
        else:
            ['resultado_inválido'], [0]
