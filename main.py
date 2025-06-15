from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import difflib

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

class Texts(BaseModel):
    text1: str
    text2: str

def extract_dependency_labels(text):
    doc = nlp(text)
    return [token.dep_ for token in doc]

def compute_syntax_similarity(text1, text2):
    deps1 = extract_dependency_labels(text1)
    deps2 = extract_dependency_labels(text2)
    matcher = difflib.SequenceMatcher(None, deps1, deps2)
    score = matcher.ratio()
    return score

@app.post("/syntax_similarity")
def syntax_similarity(data: Texts):
    score = compute_syntax_similarity(data.text1, data.text2)
    return {"syntax_similarity_score": round(score, 4)}
