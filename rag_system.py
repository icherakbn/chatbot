import os
import uuid
import logging
from typing import Dict
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langdetect import detect, DetectorFactory
from tqdm import tqdm
import time

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGChatbot")

class RAGSystem:
    def __init__(self):
        self.DATA_PATH = "qa_dataset_with_paraphrases.csv"
        self.COLLECTION_NAME = "rag_sys"
        self.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        self.QDRANT_URL = os.getenv("QDRANT_URL", "https://159c29a8-dfbd-45da-b2ce-80b6e9a5a3d6.europe-west3-0.gcp.cloud.qdrant.io")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-G7VlZUkqUsGi2-TdEZRtlfpKBjjNWJ5nKx6TKAXbxA")

        self.df = None
        self.embedder = None
        self.client = None
        self.is_initialized = False

    def initialize(self) -> bool:
        try:
            logger.info("🔧 Initializing RAG system...")
            if not os.path.exists(self.DATA_PATH):
                logger.error(f"File not found: {self.DATA_PATH}")
                return False

            # Try different encodings
            encodings = ['utf-8', 'ISO-8859-1', 'latin-1', 'cp1252']
            df_loaded = False
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.DATA_PATH, encoding=encoding)
                    logger.info(f"✅ CSV loaded with encoding: {encoding}")
                    df_loaded = True
                    break
                except Exception as e:
                    logger.info(f"Failed with encoding {encoding}: {e}")
                    continue
            
            if not df_loaded:
                logger.error("Could not read CSV with any encoding")
                return False
                
            # Check columns (case-insensitive)
            df_columns = [col.strip() for col in self.df.columns]
            logger.info(f"Found columns: {df_columns}")
            
            # Map columns to standard names if needed
            column_mapping = {}
            for col in df_columns:
                if col.lower() in ['question', 'questions']:
                    column_mapping[col] = 'Question'
                elif col.lower() in ['answer', 'answers', 'response']:
                    column_mapping[col] = 'Answer'
            
            if column_mapping:
                self.df = self.df.rename(columns=column_mapping)
                logger.info(f"Renamed columns: {column_mapping}")
            
            if not {"Question", "Answer"}.issubset(self.df.columns):
                logger.error(f"CSV must contain 'Question' and 'Answer' columns. Found: {list(self.df.columns)}")
                return False

            self.df.dropna(subset=["Question", "Answer"], inplace=True)
            self.df = self.df[self.df["Question"].str.strip() != ""]
            self.df = self.df[self.df["Answer"].str.strip() != ""]

            self.embedder = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
            logger.info("✅ Embedding model loaded.")

            self.client = QdrantClient(
                url=self.QDRANT_URL,
                api_key=self.QDRANT_API_KEY,
                timeout=60,
            )
            logger.info("🔌 Connected to Qdrant.")

            if self.COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
                test_vector = self.embedder.encode(["test"])[0]
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(size=len(test_vector), distance=Distance.COSINE)
                )
                logger.info(f"🗂️ Collection '{self.COLLECTION_NAME}' created.")

            if self.client.get_collection(self.COLLECTION_NAME).points_count == 0:
                logger.info("📥 Uploading data to Qdrant in batches...")

                batch_size = 100
                questions = self.df["Question"].tolist()
                answers = self.df["Answer"].tolist()
                vectors = self.embedder.encode(questions, show_progress_bar=True)

                total = len(vectors)
                for i in tqdm(range(0, total, batch_size)):
                    batch_points = [
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec.tolist(),
                            payload={"Question": questions[j], "Answer": answers[j]}
                        )
                        for j, vec in enumerate(vectors[i:i+batch_size], start=i)
                    ]
                    try:
                        self.client.upsert(collection_name=self.COLLECTION_NAME, points=batch_points)
                        time.sleep(0.5)  # slight pause to avoid hitting rate limits
                    except Exception as e:
                        logger.error(f"❌ Failed uploading batch {i}-{i+batch_size}: {e}")
                        continue

                logger.info(f"✅ Uploaded {total} Q&A pairs in batches.")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.exception(f"❌ Initialization failed: {e}")
            return False

    def find_best_answer(self, query: str, top_k: int = 3, min_score: float = 0.3) -> Dict:
        if not self.is_initialized:
            return {"answer": "System not initialized.", "error": True}
        try:
            query_vector = self.embedder.encode(query).tolist()
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=min_score,
            )
            if not results:
                return {"answer": "❌ No relevant answer found.", "error": False}

            best = results[0]
            # Return the answer without any prefix
            return {"answer": best.payload['Answer'], "error": False}

        except Exception as e:
            logger.exception(f"❌ Error retrieving answer: {e}")
            return {"answer": "Internal error occurred.", "error": True}

rag_system = RAGSystem()

def handle_query(query: str) -> str:
    if not rag_system.is_initialized:
        if not rag_system.initialize():
            return "❌ Failed to initialize system."
    if not query.strip():
        return "⚠️ Please enter a question."

    try:
        lang = detect(query)
    except Exception:
        return "⚠️ Could not detect language of the query."

    supported_languages = ['fr', 'en']
    if lang not in supported_languages:
        return f"⚠️ Language detected: '{lang}'. Currently only French and English are supported."

    result = rag_system.find_best_answer(query)
    return result["answer"]

# RUN THIS TO CREATE COLLECTION AND UPLOAD DATA
if __name__ == "__main__":
    print("🚀 Starting RAG System...")
    print("This will create the 'rag_sys' collection and upload your data!")
    
    # Initialize the system (creates collection and uploads data)
    if rag_system.initialize():
        print("\n🎉 SUCCESS!")
        print("✅ Collection 'rag_sys' created")
        print("✅ Data uploaded to Qdrant")
        print("✅ System is ready to use!")
        
        # Test with a sample query
        print("\n🧪 Testing with a sample query...")
        test_query = "test question"
        result = handle_query(test_query)
        print(f"Test result: {result}")
        
        # Interactive mode
        print("\n💬 You can now ask questions!")
        while True:
            user_query = input("\nEnter your question (or 'quit' to exit): ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            if user_query.strip():
                answer = handle_query(user_query)
                print(f"\nAnswer: {answer}")
    else:
        print("❌ Failed to initialize system")
        print("Make sure 'qa_dataset_with_paraphrases.csv' exists in your directory!")
