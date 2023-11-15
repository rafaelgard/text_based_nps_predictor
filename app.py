from src.projeto.nlp import nlp_predictor
from fastapi import FastAPI
from typing import Union

# if __name__ == '__main__':

nlp_object = nlp_predictor(retrain_model = False)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{comentario}")
def predict(comment: str):
    classification, probabilities = nlp_object.predict(comment)
    return {'resultado':classification.tolist(), 'probabilities':probabilities.tolist()}