from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import difflib
import os
import uvicorn

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

class Texts(BaseModel):
    text1: str
    text2: str

# 依存関係ラベルの一致度（既存）
def extract_dependency_labels(text):
    doc = nlp(text)
    return [token.dep_ for token in doc]

def compute_dependency_similarity(text1, text2):
    deps1 = extract_dependency_labels(text1)
    deps2 = extract_dependency_labels(text2)
    matcher = difflib.SequenceMatcher(None, deps1, deps2)
    return matcher.ratio()

# 文の長さ（トークン数）の類似度（1 - 差 / max）
def compute_token_length_similarity(doc1, doc2):
    len1 = len(doc1)
    len2 = len(doc2)
    return 1 - abs(len1 - len2) / max(len1, len2)

# 依存木の最大深さを計算（再帰）
def get_dependency_depth(token):
    if not list(token.children):
        return 1
    else:
        return 1 + max(get_dependency_depth(child) for child in token.children)

def compute_tree_depth_similarity(doc1, doc2):
    depth1 = max(get_dependency_depth(sent.root) for sent in doc1.sents)
    depth2 = max(get_dependency_depth(sent.root) for sent in doc2.sents)
    return 1 - abs(depth1 - depth2) / max(depth1, depth2)

# 総合スコアの計算
def compute_syntax_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    dep_sim = compute_dependency_similarity(text1, text2)
    len_sim = compute_token_length_similarity(doc1, doc2)
    depth_sim = compute_tree_depth_similarity(doc1, doc2)

　# 総合スコアの計算
    final_score = (
        0.5 * dep_sim +
        0.25 * len_sim +
        0.25 * depth_sim
    )
    return round(final_score, 4)

@app.post("/syntax_similarity")
def syntax_similarity(data: Texts):
    score = compute_syntax_similarity(data.text1, data.text2)
    return {"syntax_similarity_score": score}  # ← FileMaker用にスコアだけ返す

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
