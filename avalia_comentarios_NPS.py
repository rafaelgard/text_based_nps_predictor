'''Importação das bibliotecas'''
import pandas as pd
import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
#pip install stop_words
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

#from nltk.tokenize import word_tokenizeimport string
from nltk.corpus import stopwords
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
import difflib

def importa():

    '''Importação dos dados'''
    caminho_1 = r"F:\Projetos\1-Projeto Opiniões Raquel\Bases opiniões\Bases opiniões\202202\Compras_202202.xlsx"
    caminho_2 = r"F:\Projetos\1-Projeto Opiniões Raquel\Bases opiniões\Bases opiniões\202203\Compras_202203.xlsx"
    caminho_3 = r"F:\Projetos\1-Projeto Opiniões Raquel\Bases opiniões\Bases opiniões\202204\Compras_202204.xlsx"
    caminho_4 = r"F:\Projetos\1-Projeto Opiniões Raquel\Bases opiniões\Bases opiniões\202205\Compras_202205.xlsx"

    df1 = pd.read_excel(caminho_1,
                        sheet_name='COMPRAS', 
                        usecols=['Comentário', 'Categoria'])

    df2 = pd.read_excel(caminho_2, 
                        sheet_name='Base opiniões compras', 
                        usecols=['Comentário', 'Categoria'])

    df3 = pd.read_excel(caminho_3,
                        sheet_name='opiniões', 
                        usecols=['Comentário', 'Categoria'])

    df4 = pd.read_excel(caminho_4,
                        sheet_name='Compras', 
                        usecols=['Comentário', 'Categoria'])

    df = pd.concat([df1, df2, df3, df4])

    # df=df4

    del df1, df2, df3, df4

    df.dropna(inplace=True)
    df = df[df['Comentário'].str.isnumeric() == False]

    df['Categoria'].fillna(value='Não categorizado', inplace=True)

    df['Categoria'] = df['Categoria'].str.lower()
    df['Comentário'] = df['Comentário'].str.lower()

    # df['Categoria'] = df['Categoria'].str.replace('comentários positivo', 'elogio')
    # df['Categoria'] = df['Categoria'].str.replace('comentários positivos', 'elogio')
    # df['Categoria'] = df['Categoria'].str.replace('elogios', 'elogio')
    # df['Categoria'] = df['Categoria'].str.replace('limite baixo', 'limite')
    # df['Categoria'] = df['Categoria'].str.replace('distribuição limite', 'limite')


    #df.head()

    '''Analisando as palavras mais comuns'''
    stop_words = get_stop_words('portuguese')

    vectorizer = CountVectorizer(stop_words=stop_words)
    bag_of_words = vectorizer.fit(df['Comentário'].tolist())
    bag_of_words = vectorizer.transform(df['Comentário'].tolist())

    # print('Tamanho do vocabulário:{}' .format(len(vectorizer.vocabulary_)),'\n')
    # print('Tamanho do conteudo:\n{}' .format(vectorizer.vocabulary_))

    '''Removendo a pontuação'''
    # Esta função remove a pontuação da análise
    def remove_pontuacao(review):
        review = "".join([char for char in review if char not in string.punctuation])
        return review

    # Remove a pontuação dos reviews
    df['Comentário']=df['Comentário'].apply(lambda x: remove_pontuacao(x))


    '''Lematização dos Reviews'''
    # Esta função tokeniza a string contendo a análise
    def lematiza(review):
        porter = PorterStemmer('portuguese')
        stemmed = [porter.stem(word) for word in review]
        return stemmed

    # Remove a pontuação dos reviews
    df['Comentário']=df['Comentário'].apply(lambda x: remove_pontuacao(x))

    df.index=np.arange(len(df))

    #df.head()

    df.groupby('Categoria').count().sort_values('Comentário', ascending=False).to_excel('qtd_por_categoria.xlsx')
    #df.groupby('Categoria').count().sort_values('Comentário', ascending=False)

    #df = df[df['Categoria'].isin(['elogio', 'limite', 'anuidade', 'tarifas/ anuidade', 'loja'])]
    df.groupby('Categoria').count().sort_values('Comentário', ascending=False)

    stopWords = set(stopwords.words('portuguese'))
    stopWords = get_stop_words('portuguese')

    vectorizer = TfidfVectorizer(stop_words=stopWords)
    vectors = vectorizer.fit_transform(df['Comentário'])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    dfx = pd.DataFrame(denselist, columns=feature_names)

    dfx = dfx[dfx.isna()==False]
    dfx.head()

    # Recebe um comentário para buscar comentários similares na base

    #comentario = input('Digite um comentário: ')
    # comentário = 'pouco limite'

    # Cria a lista com todos os comentários
    comentarios = df['Comentário'].tolist()
    #print(list_of_all_movies)

    return df, comentarios, vectors

def avalia_comentarios(df, comentarios, vectors, comentario:str):

    if comentario!='':
        # print(vectors)

        print(df)

        similarity = cosine_similarity(vectors)
        print(similarity)

        print(similarity.shape)

        # find closest match for suggested movie
        find_close_match = difflib.get_close_matches(comentario, comentarios)
        print(find_close_match)

        close_match = find_close_match[0]
        print(close_match)

        # find index of the movie with title

        index_of_movie = df['Comentário'].loc[df.Comentário == close_match].index[0]
        print(index_of_movie)

        # getting list of similar movies

        similarity_score = list(enumerate(similarity[index_of_movie]))
        print(similarity_score)

        len(similarity_score)

        # sorting the movies based on their similarity scores

        sorted_similar= sorted(similarity_score, key = lambda x:x[1], reverse = True)
        print(sorted_similar)

        # print the name of similar movies based on the index

        # print the name of similar movies based on the index
        # print('comentário:', comentario, '\n')
        # print('Comentários semelhantes : \n')
        
        comentarios = []

        print('df.index')
        print(df.index)

        for coment in sorted_similar:
            title_from_index = df[df.index == comentario[0]]['Comentário'].values[0]
            comentarios.append(title_from_index)
            # if(i < 30):
            #     print(i,'.',title_from_index)
            #     i+=1

        return comentarios

