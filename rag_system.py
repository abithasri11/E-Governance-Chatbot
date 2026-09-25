import os
import json
import faiss
from sentence_transformers import SentenceTransformer


class RAGSystem:
    def __init__(self, data_path):
        self.data_path = data_path

        with open(
            os.path.join(data_path, "rag_documents.json"),
            "r",
            encoding="utf-8"
        ) as f:
            self.documents = json.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = faiss.read_index(
            os.path.join(data_path, "faiss_index.bin")
        )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                f"FAISS index contains {self.index.ntotal} vectors, "
                f"but rag_documents.json contains {len(self.documents)} documents. "
                "Rebuild the FAISS index using the same document order."
            )

    def find_service(self, question):
        question = question.lower()

        for document in self.documents:
            service = document.get("metadata", {}).get(
                "service_name", ""
            ).lower()

            if service and service in question:
                return document

        return None

    def retrieve(self, question, intent):
        service = self.find_service(question)

        if service:
            return service

        search_text = question + " " + intent

        vector = self.model.encode(
            [search_text],
            convert_to_numpy=True
        )

        distances, ids = self.index.search(vector, 1)
        return self.documents[int(ids[0][0])]
