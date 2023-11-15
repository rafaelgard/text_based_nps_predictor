from src.projeto.nlp import nlp_predictor

if __name__ == '__main__':

    new_comments = ["Este serviço é excelente!",
                "Estou insatisfeito com o atendimento e o produto é horrível",
                "Péssimo cartão! Não fui bem atendida e meu limite é baixo"
                ]

    # caso queira treinar o modelo
    nlp_object = nlp_predictor(retrain_model = False)
    nlp_object.predict(new_comments)
    