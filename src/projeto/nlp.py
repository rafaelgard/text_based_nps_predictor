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
from pyod.models.ecod import ECOD
from scipy.sparse import csr_matrix
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import lightgbm as lgb


class nlp_predictor:
    def __init__(self, retrain_model=False):
        '''Classe que permite a utilização dos métodos implementados'''
        # se retrain_model=True, o modelo é retreinado,
        # caso contrário é carregado um modelo já treinado
        self.retrain_model = retrain_model
        self.model = None
        self.vectorizer = None
        self.model_type = 'RNN'
        self.initial_setup()

    def initial_setup(self):
        '''Executa as configurações iniciais da classe'''
        self.model, self.vectorizer = self.get_model()

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

        breakpoint()

        # separando os dados em conjunto de teste e treino
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=30, stratify=y)

        # instanciando e treinando o CountVectorizer
        stopWords = nltk.corpus.stopwords.words('portuguese')
        vectorizer = CountVectorizer(analyzer="word", stop_words=stopWords)
        X_train_vectorizer = vectorizer.fit_transform(X_train)
        X_test_vectorizer = vectorizer.transform(X_test)

        # print('X_train_vectorizer')
        # print(X_train_vectorizer)
        
        # X_train_vectorizer_sem_outiliers, y_train_sem_outiliers = self.remove_outiliers(X_train_vectorizer, y_train)
        # breakpoint()    
       
        # instanciando o modelo
        # model = MultinomialNB()
        
        import lightgbm as lgb
        model = lgb.LGBMClassifier()#colsample_bytree=0.8

        # treinando o modelo nos dados de treinamento
        # model.fit(X_train_vectorizer_sem_outiliers, y_train_sem_outiliers)
        model.fit(X_train_vectorizer.toarray(), np.array(y_train))
        y_pred = model.predict(X_test_vectorizer.toarray())
        # accuracy=accuracy_score(y_pred, y_train_sem_outiliers)
        accuracy=accuracy_score(np.array(y_test),y_pred)
        print('Training-set accuracy score: {0:0.4f}'. format(accuracy))
        # print(classification_report(y_train_sem_outiliers, y_pred))
        print(classification_report(np.array(y_test),y_pred))
        
        # salvando o modelo
        # dump((model, X_train_vectorizer_sem_outiliers, vectorizer), 'src/models/new_model.joblib')

        # fazendo as previsões no conjunto de teste
        # y_pred = model.predict(X_test_vectorizer)

        # avaliando o desempenho do modelo
        # accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred)

        # print(f'\nAcurácia: {accuracy:.2f}')
        # print(f'\nRelatório de Classificação:\n', report)

        cross_val_score_result = cross_val_score(
            estimator=model, X=X_test_vectorizer.toarray(), y=np.array(y_test), cv=5, n_jobs=-1)

        print('Resultado da validação cruzada')
        print(cross_val_score_result, '\n')

        print(
            f'Média do resultado da validação cruzada: {round(cross_val_score_result.mean(), 2)}\n')

        breakpoint()	
        return model, vectorizer

    def make_model_RNN(self):
        '''Importação dos dados'''
        df = pd.read_csv(r'src/database/dataframe_final.csv')

        # dropa os dados nulos
        df.dropna(inplace=True)

        # # transforma todos os comentários para minúsculo
        # df['Comentário'] = df['Comentário'].str.lower()

        # # corrige a gramática
        # df['Comentário'] = self.corrige_gramatica(df['Comentário'])

        # # Remove a pontuação dos reviews
        # df['Comentário'] = df['Comentário'].apply(self.remove_pontuacao)

        # # lematiza a coluna comentário
        # # Carregando o modelo em português do Brasil do spaCy
        # nlp = spacy.load("pt_core_news_sm")

        # df['Comentário'] = df['Comentário'].apply(
        #     lambda x: " ".join(self.lemmatize_words(x, nlp)))

        # df.index = np.arange(df.shape[0])

        df['TARGET'] = df['Nota'].apply(lambda x: self.avalia_nota(x))

        # df = df[df['Comentário'].isna() == False]

        # df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

        # # Dividindo os dados em conjunto de treinamento e conjunto de teste
        
        # Pré-processamento dos dados
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        df['tokenized_comments'] = df['Comentário'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Converta categorias em rótulos numéricos
        category_mapping = {'Promotor': 0, 'Neutro': 1, 'Detrator': 2}
        df['label'] = df['TARGET'].map(category_mapping)


        x = df['Comentário'].astype('str')
        y = df['label']

        #['Comentário', 'Nota', 'TARGET', 'tokenized_comments', 'label']


        # breakpoint()
        # Separar os dados em treino e teste
        train_df, test_df = train_test_split(df[['Comentário','label']], test_size=0.2, random_state=42)

        print('Bert')
        # Obter embeddings BERT para os comentários
        bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        def get_bert_embedding(comment):
            input_ids = torch.tensor([tokenizer.encode(comment, add_special_tokens=True)])
            return bert_model(input_ids)[1].detach().numpy()[0]

        train_df['bert_embeddings'] = train_df['Comentário'].apply(get_bert_embedding)
        test_df['bert_embeddings'] = test_df['Comentário'].apply(get_bert_embedding)


        # Verifique as dimensões dos embeddings
        print(f"Dimensões dos embeddings de treino: {train_df['bert_embeddings'].apply(lambda x: x.shape).unique()}")
        print(f"Dimensões dos embeddings de teste: {test_df['bert_embeddings'].apply(lambda x: x.shape).unique()}")


        train_embeddings_array = np.array(list(train_df['bert_embeddings']))
        train_data = lgb.Dataset(train_embeddings_array, label=train_df['label'])

        # Treinamento do modelo LightGBM
        params = {'objective': 'multiclass', 'num_class': 3}
        model = lgb.train(params, train_data, num_boost_round=100)

        # Previsões no conjunto de teste
        test_predictions = model.predict(list(test_df['bert_embeddings'])).argmax(axis=1)

        # Avaliação do modelo
        accuracy = accuracy_score(test_df['label'], test_predictions)
        print(f'Acurácia do modelo: {accuracy}')

        # Generate a classification report
        report = classification_report(test_df['label'], test_predictions)
        print('Classification report:\n', report)

        breakpoint()

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

    def remove_outiliers(self, X_train, y_train):

        y_train = y_train.reset_index(drop=True)
        print('iniciando o remove outiliers')
        clf = ECOD()
        clf.fit(X_train.toarray())
        # print('predict')
        # out = clf.predict(X_train.toarray())

        X_train = X_train.toarray()
        y_train = np.array(y_train)

        out = clf.fit_predict(X_train)

        df = pd.DataFrame(X_train)

        df['outliers'] = out

        df['index'] = df.index

        df_sem_outliers = df[df['outliers']==0].copy()

        df_sem_outliers.drop(['outliers'], axis=1, inplace=True)

        # breakpoint()
        y_train_sem_outliers = y_train[df_sem_outliers['index']]

        X_train_sem_outliers = df_sem_outliers.to_numpy(copy=True)

        del df, df_sem_outliers

        # X_train_sem_outliers = csr_matrix(X_train_sem_outliers)

        # breakpoint()

        # print('scores')
        # # get outlier scores
        # y_train_scores = clf.decision_scores_  # raw outlier scores on the train data
        # y_test_scores = clf.decision_function(X_test.toarray())  # predict raw outlier scores on test

        # print('y_train_scores')
        # print(y_train_scores)
        # print(y_test_scores)
        # print(y_test_scores)
        # # breakpoint()
        print('Finalizou remove outliers')
        return X_train_sem_outliers, y_train_sem_outliers

    def classifica_nps(self, data) -> (np.array, np.array):
        "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

        #todo: caso eu altere os modelos essas funções podem ter que mudar
        comentarios_tfidf = self.vectorizer.transform(data)

        probabilidades = self.model.predict_proba(comentarios_tfidf)
        resultado = self.model.predict(comentarios_tfidf)

        return resultado, probabilidades

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
            if self.model_type == 'RNN':
                model, tokenizer, label_encoder = self.make_model_RNN()
            else:
                model, vectorizer = self.make_model()

        else:
            if self.model_type == 'RNN':
                pass

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

            resultado, probabilidades = self.classifica_nps(comentario)

            return resultado, probabilidades
        
        else:
            ['resultado_inválido'], [0]


nlp_obj = nlp_predictor(True)

data =  ["Este serviço é excelente!",
         "Estou insatisfeito com o atendimento e o produto é horrível",
         "Péssimo cartão! Não fui bem atendida e meu limite é baixo"]

nlp_obj.predict(data)
breakpoint()