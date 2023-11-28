import streamlit as st
from projeto import nlp_predictor

nlp = nlp_predictor(model_type='', retrain_model=False)

st.title("Modelo de Previsão de NPS")

col1, col2 = st.columns(2)
with col1:

    # 'Comentário', 'Digite o comentário'
    comentario = st.text_input(
        label='', value="Digite o comentário", label_visibility='hidden')

calcular = False

with col2:
    st.write('')
    st.write('')
    calcular = st.button("Calcular NPS", type="primary")

if calcular:
    if comentario in ('Digite o comentário', ''):
        st.write('Digite um comentário válido!')

    elif comentario not in ('Digite o comentário', ''):

        resultado, probabilities = nlp.predict(comentario)

        with st.spinner('Analisando...'):

            prob_detrator = str(
                round(probabilities[0][0].astype(float)*100, 2))+"%"
            prob_neutro = str(
                round(probabilities[0][1].astype(float)*100, 2))+"%"
            prob_promotor = str(
                round(probabilities[0][2].astype(float)*100, 2))+"%"

            probabilidades = {'Detrator': prob_detrator,
                              'Neutro': prob_neutro, 'Promotor': prob_promotor}
            cores = {'Detrator': "Red",
                     'Neutro': "inverse", 'Promotor': "Blue"}

            col3, col4 = st.columns(2)

            if resultado[0] != '':
                col3.metric("Resultado final:", resultado[0])
