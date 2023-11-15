import streamlit as st
import pandas as pd
from projeto import nlp_predictor
 
nlp = nlp_predictor(False)
model, X_train_vectorizer, vectorizer = nlp.load_model()

st.title("Modelo de Previsão de NPS")

col1, col2 = st.columns(2)
with col1:

    comentario = st.text_input(label='', value="Digite o comentário", label_visibility= 'hidden')#'Comentário', 'Digite o comentário'

calcular=False

with col2:
    st.write('')
    st.write('')
    calcular = st.button("Calcular NPS", type="primary")

if calcular:
    
    if comentario == "Digite o comentário" or comentario == "":
        st.write('Digite um comentário válido!')

    elif comentario != "Digite o comentário" and comentario != "":

        resultado, probabilities = nlp.predict(comentario)

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
 
