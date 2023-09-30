import streamlit as st
import pandas as pd
import time
# from avalia_comentarios_NPS import avalia_comentarios, importa
from projeto import classifica_nps, load_model
# from streamlit_image_select import image_select
# from recomendation_system import recomenda_itens, itens_iniciais
#  from utils import return_images_path

# def return_images_path(itens_escolhidos:list):
#     '''Retorna o caminho das imagens dos itens escolhidos'''

#     images_path = {
#                     "ARROZ": "app_images/arroz.png",
#                     "FEIJAO CARIOCA": "app_images/feijao_carioca.png",
#                     "AÇÚCAR": "app_images/acucar.png",
#                     "CAFÉ": "app_images/cafe.png",
#                     "FLOCÃO": "app_images/flocao.png",
#                     "BISCOITO": "app_images/biscoito.png",
#                     "LEITE": "app_images/leite.png"                   
#                    }

#     paths = []

#     for item in itens_escolhidos:
#         paths.append(images_path.get(item))
#         print(images_path.get(item))

#     return paths
      

# st.set_page_config(
#     page_title="Sistema de Recomendação de Compras",
#     page_icon="🧊",
#     layout="centered",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     }
# )      
model, X_train_vectorizer, vectorizer = load_model()

st.title("Modelo de Previsão de NPS")

col1, col2 = st.columns(2)
with col1:

    comentario = st.text_input(label='', value="Digite o comentário", label_visibility= 'hidden')#'Comentário', 'Digite o comentário'

calcular=False

with col2:
    # st.button("Calcular NPS", type="primary")
    st.write('')
    st.write('')
    calcular = st.button("Calcular NPS", type="primary")

if calcular:
    
    if comentario == "Digite o comentário" or comentario == "":
        st.write('Digite um comentário válido!')

    elif comentario != "Digite o comentário" and comentario != "":

        # progress_text = "Operation in progress. Please wait."
        # my_bar = st.progress(0, text=progress_text)
        resultado, probabilities = classifica_nps(model, vectorizer, comentario)

        # for percent_complete in range(100):
        #     time.sleep(0.01)
        #     my_bar.progress(value=percent_complete + 1, text=progress_text)
        with st.spinner('Analisando...'):

            prob_detrator = str(round(probabilities[0][0].astype(float)*100,2))+"%"
            prob_neutro =  str(round(probabilities[0][1].astype(float)*100,2))+"%"
            prob_promotor =  str(round(probabilities[0][2].astype(float)*100,2))+"%"

            probabilidades = {'Detrator':prob_detrator, 'Neutro':prob_neutro, 'Promotor':prob_promotor}
            cores = {'Detrator':"Red", 'Neutro':"inverse", 'Promotor':"Blue"}
            # print('probabilidades')delta_color="inverse"
            # print(probabilidades)

            # time.sleep(3)
       
            col3, col4 = st.columns(2)

            if resultado[0] != '':
                col3.metric("Resultado final:", resultado[0])
 
