from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
from bertopic import BERTopic
import uvicorn

app = FastAPI(title="BERTopic Fake News Analyzer")

print("Loading model...")
topic_model = BERTopic.load("model/bertopic_fake_model")
print("Model has been loaded!")

class TextInput(BaseModel):
    text:str

@app.post("/predict")
def predict(input: TextInput):
    topics, probs = topic_model.transform([input.text])
    
    topic_id = int(topics[0])
    
    if topic_id == -1:
        topic_name = "outlier (не попало в тему)"
        keywords = []
    else:
        topic_name = f"Topic {topic_id}"
        keywords = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
    
    result = {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "probability": float(probs[0].max()) if probs.shape[1] > 1 else 1.0,
        "top_keywords": keywords
    }
    return result

@app.get("/")
def root():
    return {"Bertopic is ready, send a request on /predict endpoint!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
