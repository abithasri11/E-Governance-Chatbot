class EGovernanceChatbot:
    def __init__(self, ml, rag):
        self.ml = ml
        self.rag = rag

    def answer(self, question):
        intent, confidence = self.ml.predict(question)

        document = self.rag.retrieve(
            question,
            intent
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "document": document
        }
