import faiss
from sentence_transformers import SentenceTransformer
import time
import json

class SemanticCache:
    def __init__(self, json_file="cache_file.json", threshold=0.35):
        self.index, self.encoder = self.init_cache()
        self.euclidean_threshold = threshold
        self.json_file = json_file
        self.cache = self.retrieve_cache(self.json_file)

    def init_cache(self):
        index = faiss.IndexFlatL2(768)
        encoder = SentenceTransformer("all-mpnet-base-v2")
        return index, encoder

    def retrieve_cache(self, json_file):
        try:
            with open(json_file, "r") as file:
                cache = json.load(file)
        except FileNotFoundError:
            cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}
        return cache

    def store_cache(self, json_file, cache):
        with open(json_file, "w") as file:
            json.dump(cache, file)

    def ask(self, question, query_database_func):
        start_time = time.time()
        try:
            embedding = self.encoder.encode([question])
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print("Answer recovered from Cache.")
                    print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                    print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")
                    print(f"response_text: " + self.cache["response_text"][row_id])
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache["response_text"][row_id]

            answer = query_database_func([question])
            response_text = answer["documents"][0][0]
            print("Answer recovered from ChromaDB.")
            print(f"response_text: {response_text}")

            self.update_cache(question, embedding, answer, response_text)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")

    def update_cache(self, question, embedding, answer, response_text):
        self.cache["questions"].append(question)
        self.cache["embeddings"].append(embedding[0].tolist())
        self.cache["answers"].append(answer)
        self.cache["response_text"].append(response_text)
        self.index.add(embedding)
        self.store_cache(self.json_file, self.cache)
