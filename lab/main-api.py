from typing import Union
from modelscope import AutoModel

from fastapi import FastAPI
import json
from pydantic import BaseModel

app = FastAPI()



JinNaAI_Model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method


# cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

# def calculate_similarity(sentence1, sentence2):
#     JinNaAI_Model = get_sts_model()
#     embeddings = JinNaAI_Model.encode([sentence1, sentence2])
#     return cos_sim(embeddings[0], embeddings[1])


class Item(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/encode")
def encode( item: Item):
    if item.text:
        embeddings = JinNaAI_Model.encode([item.text])
        return json.dumps(embeddings[0].tolist())
    return ""
    

@app.get("/encode")
def encode(text: str):
    if text:
        embeddings = JinNaAI_Model.encode([text])
        return json.dumps(embeddings[0].tolist())
    return ""
    


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}