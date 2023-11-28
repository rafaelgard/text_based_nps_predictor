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
from transformers import BertForSequenceClassification, AdamW, BertConfig

from torch import cuda

class nlp_predictor:
    def __init__(self, model_type = 'BERT', retrain_model=False):
        '''Classe que permite a utilização dos métodos implementados'''
        # se retrain_model=True, o modelo é retreinado,
        # caso contrário é carregado um modelo já treinado
        self.retrain_model = retrain_model
        self.model = None
        self.nlp_model = None
        self.vectorizer = None
        self.tokenizer = None
        self.model_type = model_type
        self.initial_setup()

    def initial_setup(self):
        '''Executa as configurações iniciais da classe'''

        if self.model_type != 'BERT':
            self.model, self.vectorizer = self.get_model()

        elif self.model_type == 'BERT':
            self.model, self.tokenizer = self.get_model()
            self.nlp_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    def pre_processamento(self, dataframe):
        
        # transforma todos os comentários para minúsculo
        dataframe['Comentário'] = dataframe['Comentário'].str.lower()

        # corrige a gramática
        dataframe['Comentário'] = self.corrige_gramatica(dataframe['Comentário'])

        # Remove a pontuação dos reviews
        dataframe['Comentário'] = dataframe['Comentário'].apply(self.remove_pontuacao)

        # lematiza a coluna comentário
        # Carregando o modelo em português do Brasil do spaCy
        nlp = spacy.load("pt_core_news_sm")

        dataframe['Comentário'] = dataframe['Comentário'].apply(
            lambda x: " ".join(self.lemmatize_words(x, nlp)))

        dataframe.index = np.arange(dataframe.shape[0])

        dataframe = dataframe[dataframe['Comentário'].isna() == False]

        dataframe.drop_duplicates(keep='last', inplace=True, ignore_index=True)

        return dataframe


    def make_model(self):
        '''Treina o modelo nltk e MultinomialNB'''
        df = pd.read_csv(r'src/database/dataframe_final.csv')

        # dropa os dados nulos
        df.dropna(inplace=True)

        # df = self.pre_processamento(df)

        df['TARGET'] = df['Nota'].apply(lambda x: self.avalia_nota(x))

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

        # print('X_train_vectorizer')
        # print(X_train_vectorizer)
        
        # X_train_vectorizer_sem_outiliers, y_train_sem_outiliers = self.remove_outiliers(X_train_vectorizer, y_train)
       
        # instanciando o modelo
        model = MultinomialNB()
        # model = lgb.LGBMClassifier()#colsample_bytree=0.8

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
        dump((model, vectorizer), 'src/models/model.joblib')

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

        # breakpoint()	
        return model, vectorizer

    def make_model_bert(self):
        '''Treina o modelo Bert e lgb'''

        df = pd.read_csv(r'src/database/dataframe_final.csv')

        # df = self.pre_processamento(df)

        # dropa os dados nulos
        df.dropna(inplace=True)

        df['TARGET'] = df['Nota'].apply(lambda x: self.avalia_nota(x))

        # tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', max_length=512)
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=True)

        df['tokenized_comments'] = df['Comentário'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        # Converta categorias em rótulos numéricos
        # category_mapping = {'Promotor': 0, 'Neutro': 1, 'Detrator': 2}

        # id2label = {0:'Promotor', 1:'Neutro', 2:'Detrator'}
        label2id = {'Promotor': 0, 'Neutro': 1, 'Detrator': 2}

        # num_labels = len(id2label.keys())

        df['label'] = df['TARGET'].map(label2id)

        print('Bert')
        # Obter embeddings BERT para os comentários
        # bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

        # model = BertForSequenceClassification.from_pretrained(
        #             "neuralmind/bert-base-portuguese-cased", 
        #             num_labels = num_labels, 
        #             id2label=id2label, 
        #             label2id=label2id, 
        #             output_attentions=False,
        #             output_hidden_states=False)
        
        # bert_model = BertForSequenceClassification.from_pretrained(
        #             "neuralmind/bert-base-portuguese-cased", 
        #             num_labels = num_labels, 
        #             id2label=id2label, 
        #             label2id=label2id)
        
        # if cuda.is_available():
        #     device = 'cuda'
        # else: 
        #     device = 'cpu'

        # print(f'device:{device}')
        
        # model.to(device)

        # Separar os dados em treino e teste
        train_df, test_df = train_test_split(df[['Comentário','label']], 
                                             test_size=0.3, 
                                             random_state=30, 
                                             stratify=df['label'])

        def get_bert_embedding(comment):
            input_ids = torch.tensor([tokenizer.encode(comment, add_special_tokens=True)])
            return bert_model(input_ids)[1].detach().numpy()[0]
            # return model(input_ids)[1].detach().numpy()[0]

        train_df['bert_embeddings'] = train_df['Comentário'].apply(get_bert_embedding)
        test_df['bert_embeddings'] = test_df['Comentário'].apply(get_bert_embedding)

        # Verifique as dimensões dos embeddings
        print(f"Dimensões dos embeddings de treino: {train_df['bert_embeddings'].apply(lambda x: x.shape).unique()}")
        print(f"Dimensões dos embeddings de teste: {test_df['bert_embeddings'].apply(lambda x: x.shape).unique()}")

        train_embeddings_array = np.array(list(train_df['bert_embeddings']))
        
        X_train = train_embeddings_array
        y_train = train_df['label']

        # print('Iniciando remoção dos outiliers')
        # X_train, y_train = self.remove_outiliers(X_train, y_train)
        # print('Finalizou remoção dos outiliers')

        # Pré-processamento dos dados
        # from imblearn.under_sampling import ClusterCentroids
        # from collections import Counter
        # cc = ClusterCentroids(random_state=0)
        # X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
        # print(sorted(Counter(y_resampled).items()))

        # from imblearn.combine import SMOTEENN
        # smote_enn = SMOTEENN(random_state=0)
        # X_train, y_train = smote_enn.fit_resample(X_train, y_train)

        train_data = lgb.Dataset(X_train, y_train)

        print('Iniciou o treinamento do modelo')
        # Treinamento do modelo LightGBM
        params = {'objective': 'multiclass', 'num_class': 3}
        model = lgb.train(params, train_data, num_boost_round=100)

        print('Finalizou o treinamento do modelo')

        # salvando o modelo
        dump((model, tokenizer), 'src/models/bert_model.joblib')
        print('Modelo bert salvo!')
        # breakpoint()

        # Previsões no conjunto de teste
        test_predictions = model.predict(list(test_df['bert_embeddings'])).argmax(axis=1)

        # Avaliação do modelo
        accuracy = accuracy_score(test_df['label'], test_predictions)
        print(f'Acurácia do modelo: {accuracy}')

        # Generate a classification report
        report = classification_report(test_df['label'], test_predictions)
        print('Classification report:\n', report)

        return model, tokenizer


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
        '''
        Remove outiliers dos dados de treino

        X_train: np.array
        y_train: np.array

        '''

        print(f'X_train.shape:{X_train.shape}')
        print(f'y_train.shape:{y_train.shape}')

        y_train = y_train.reset_index(drop=True)

        print('iniciando o remove outiliers')
        clf = ECOD()
        clf.fit(X_train)
 
        y_train = np.array(y_train)

        outliers = clf.fit_predict(X_train)

        # cria um dataframe pandas
        df = pd.DataFrame(X_train)

        df['outliers'] = outliers

        df['index'] = df.index

        df_sem_outliers = df[df['outliers']==0].copy()

        linhas = df_sem_outliers['index']

        df_sem_outliers.drop(['outliers', 'index'], axis=1, inplace=True)

        y_train_sem_outliers = y_train[linhas]

        X_train_sem_outliers = df_sem_outliers.to_numpy(copy=True)

        print('Finalizou remove outliers')

        return X_train_sem_outliers, y_train_sem_outliers

    def get_bert_embedding(self, comment):
        input_ids = torch.tensor([self.tokenizer.encode(comment, add_special_tokens=True)])
        return self.nlp_model(input_ids)[1].detach().numpy()[0]
    
    def classifica_nps(self, data) -> (np.array, np.array):
        "Classifica o nps baseado em um ou mais comentários e retorna o resultado"

        if self.model_type != 'BERT':
        
            comentarios_tfidf = self.vectorizer.transform(data)
            probabilidades = self.model.predict_proba(comentarios_tfidf)
            resultado = self.model.predict(comentarios_tfidf)

            return resultado, probabilidades
        
        elif self.model_type == 'BERT':

            data_embeddings = [self.get_bert_embedding(x) for x in data]

            predictions = self.model.predict(data_embeddings, num_iteration=self.model.best_iteration)

            resultados = predictions.argmax(axis=1)
            
            id2label = {0:'Promotor', 1:'Neutro', 2:'Detrator'}
    
            resultados = [id2label[x] for x in resultados]

            return resultados, predictions

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

        if self.model_type != 'BERT':
            model, vectorizer = load(
                r'src\models\model.joblib')
        
            return model, vectorizer
    
        elif self.model_type == 'BERT':
            model, tokenizer = load(
            r'src\models\bert_model.joblib')
            
            return model, tokenizer

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
            if self.model_type == 'BERT':
                model, tokenizer = self.make_model_bert()
                return model, tokenizer

            else:
                model, vectorizer = self.make_model()
                return model, vectorizer

        else:
            if self.model_type == 'BERT':
                model, tokenizer = self.load_model()
                return model, tokenizer

            else:
                model, vectorizer = self.load_model()
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


nlp_obj = nlp_predictor(model_type = 'x', retrain_model=False)


# novos comentários que serão utilizados para testar o modelo gerado
novos_comentários = ["Este serviço é excelente!",
                     "Estou insatisfeito com o atendimento e o produto é horrível",
                     "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                     ]

resultado, probabilidades = nlp_obj.predict(novos_comentários)
for index, comment in enumerate(novos_comentários):
    print(f'Comentário: {novos_comentários[index]}')
    print(f'Resultado previsto: {resultado[index]}\n')

# breakpoint()