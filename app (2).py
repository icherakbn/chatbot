from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import wikipediaapi
from datasets import load_dataset

import re
import string

from word2number import w2n

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from rank_bm25 import BM25Okapi
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # ou ["*"] en dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------- FastAPI app ------------
import psycopg2

def save_chat_to_db(question: str, answer: str):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="mysecret",   # Or 'Cmc@2023' depending on what you're using
            host="rag-postgres",   # 👈 use 'rag-postgres' if calling from inside Docker
            port="5432"
        )

        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO chat_history (question, answer) VALUES (%s, %s)",
            (question, answer)
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Failed to save chat to DB: {e}")


class QueryRequest(BaseModel):
    query: str


class AnswerRequest(BaseModel):
    question: str
    context: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/search")
def search(request: QueryRequest):
    normalized_query = normalize(request.query)
    docs = hybrid_retrieve_and_rerank(normalized_query)
    return {"documents": docs}

@app.post("/generate")
def generate(request: AnswerRequest):
    answer = generate_answer(request.question, request.context)
    save_chat_to_db(request.question, answer)
    return {"answer": answer}

# ----------- Helper functions ------------

def normalize(text):
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        return str(w2n.word_to_num(text))
    except:
        return text

def exact_match_score(prediction, ground_truth):
    pred_tokens = set(normalize(prediction).split())
    true_tokens = set(normalize(ground_truth).split())
    return int(len(pred_tokens & true_tokens) > 0)

def infer_answer_type(question):
    q = question.lower()
    if "when" in q:
        return "a date"
    elif "how many" in q or "number" in q:
        return "a number"
    elif "who" in q:
        return "a person"
    elif "where" in q:
        return "a place"
    else:
        return "a short phrase"

# ----------- Initialize APIs and Models ------------

wiki = wikipediaapi.Wikipedia(
    user_agent='My Wikipedia Application (your_email@example.com)',
    language='en'
)

ds = load_dataset("facebook/kilt_tasks", "nq", streaming=False)
train_data = ds['train']

simplified_dataset = []
for item in train_data:
    if len(item['output']) > 0 and 'provenance' in item['output'][0] and len(item['output'][0]['provenance']) > 0:
        simplified_dataset.append({
            'question': item['input'],
            'answer': item['output'][0]['answer'],
            'evidence_title': item['output'][0]['provenance'][0]['title']
        })
wikipedia_content_cache = {}

def get_wikipedia_content(title: str) -> str:
    """Fetch Wikipedia content with error handling and caching."""
    if title in wikipedia_content_cache:
        return wikipedia_content_cache[title]

    try:
        page = wiki.page(title)
        if page.exists():
            content = page.text
            wikipedia_content_cache[title] = content
            return content
        else:
            return ""
    except Exception as e:
        print(f"Error fetching Wikipedia content for {title}: {e}")
        return ""

# Filter and prepare documents
sample_size = 100  # Reduced sample size
eval_titles = set(item['evidence_title'] for item in simplified_dataset[:sample_size])
filtered_docs = [item for item in simplified_dataset if item['evidence_title'] in eval_titles]

documents = [get_wikipedia_content(item['evidence_title']) for item in filtered_docs]

titles = [item['evidence_title'] for item in filtered_docs]

# BM25 Setup
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_bm25(query, top_k=5):  # Reduced top_k
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    return [documents[i] for i in top_indices], [bm25_scores[i] for i in top_indices]

# Sentence-BERT and Generator
retriever_model = SentenceTransformer("all-mpnet-base-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)

# Qdrant Client Setup
qdrant_url = "https://84b4d67b-beda-4ff8-b3c3-db566b8e0fd6.europe-west3-0.gcp.cloud.qdrant.io"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2J5_g-2vTSsgLIlwlNboPZli0zFLD2oHSQioHuxgOa8"

client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)  # Increased timeout
collection_name = "wikipedia_data"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

# Cross-encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Generate embeddings for documents and upsert to Qdrant
embeddings = []
batch_size = 50  # Reduced batch size for embeddings
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    embeddings.extend(retriever_model.encode(batch, show_progress_bar=True))

def upsert_filtered_documents(batch_size=50):  # Increased batch size
    num_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(documents))

        batch_documents = documents[start_index:end_index]
        batch_embeddings = embeddings[start_index:end_index]
        batch_titles = titles[start_index:end_index]

        points = [
            {
                'id': i + start_index,
                'vector': embedding.tolist(),
                'payload': {
                    'text': doc,
                    'title': batch_titles[i]
                }
            }
            for i, (embedding, doc) in enumerate(zip(batch_embeddings, batch_documents))
        ]

        retry_count = 0
        backoff = 5  # Start with 5 seconds
        while retry_count < 5:  # Retry up to 5 times
            try:
                client.upsert(collection_name=collection_name, points=points)
                print(f"Upserted batch {batch_index + 1}/{num_batches}")
                break
            except Exception as e:
                print(f"Error upserting batch {batch_index + 1}: {e}. Retrying in {backoff} seconds...")
                retry_count += 1
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
        else:
            print(f"Failed to upsert batch {batch_index + 1} after 5 retries.")

    print(f"Successfully attempted upsert for {len(documents)} documents.")

# Run once at startup to populate Qdrant
upsert_filtered_documents(batch_size=25)

# ----------- Retrieval & Generation functions ------------
def hybrid_retrieve_and_rerank(query, top_k=15, rerank_top_k=10, dense_weight=0.7, sparse_weight=0.3):
    normalized_query = normalize(query)  # Normalize the query
    query_embedding = retriever_model.encode([normalized_query], show_progress_bar=False)

    # Qdrant dense search
    dense_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding[0].tolist(),
        limit=top_k
    )
    dense_docs = [res.payload['text'] for res in dense_results]
    dense_scores = [res.score for res in dense_results]

    # BM25 sparse search
    sparse_docs, sparse_scores = retrieve_bm25(normalized_query, top_k=top_k)

    # Combine dense and sparse scores
    score_dict = {}
    for doc, score in zip(dense_docs, dense_scores):
        score_dict[doc] = score * dense_weight
    for doc, score in zip(sparse_docs, sparse_scores):
        score_dict[doc] = score_dict.get(doc, 0) + score * sparse_weight

    # Sort and rerank
    combined_docs_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in combined_docs_scores[:rerank_top_k]]

    # Rerank using the cross-encoder
    pairs = [[query, doc] for doc in top_docs]
    rerank_scores = reranker.predict(pairs, batch_size=8)  # Adjust batch size for efficiency
    reranked = sorted(zip(top_docs, rerank_scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked]

def generate_answer(question, context):
    # Handle greetings explicitly
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if question.lower() in greetings:
        return "Hello! How can I assist you today?"

    # Infer the type of answer expected
    answer_type = infer_answer_type(question)

    # Truncate context to fit model token limit (~500 words for flan-t5-base)
    MAX_CONTEXT_TOKENS = 500
    context_words = context.split()
    if len(context_words) > MAX_CONTEXT_TOKENS:
        # Keep beginning and end (good for coverage)
        first_part = context_words[:250]
        last_part = context_words[-250:]
        context = " ".join(first_part + ["..."] + last_part)

    prompt = (
        f"Use the context below to answer the question concisely.\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer:"
    )


    # Generate the response using the model
    result = generator(prompt, max_length=200, num_beams=4, early_stopping=True)

    # Return the generated text
    return result[0]['generated_text']