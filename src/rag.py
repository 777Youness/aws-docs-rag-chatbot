import chromadb
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

from src.embedding import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from src.utils import LLM_MODEL, get_llm_client


# Max embedding distance (Chroma default metric: squared L2) for a chunk to count as relevant.
# On this corpus, real questions score ~0.3-0.75 while greetings and off-topic text score >0.9.
MAX_DISTANCE = 0.8


class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.llm_client = get_llm_client()
        self.last_sources = []

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AWS (Amazon Web Services) assistant. You can help with Amazon S3, AWS IAM and Amazon VPC.
If the user greets you or makes small talk (hello, how are you, thanks...), reply naturally and briefly, and mention what you can help with. Ignore the context in that case.
For any technical question, answer using only the context below, which comes from the official AWS documentation.
If the context does not contain the answer, say so clearly instead of making something up.
Answer in the language of the question.

{context}"""),
            ("user", "{question}"),
        ])

    def retrieve(self, query, k=5):
        """Return the k chunks closest to the question, formatted as a context string.

        Chunks farther than MAX_DISTANCE are dropped, so greetings and off-topic
        messages get no context (and no sources).
        """
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        kept = [
            (text, meta.get("source", "unknown"))
            for text, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
            if dist <= MAX_DISTANCE
        ]
        self.last_sources = [source for _, source in kept]

        if not kept:
            return "(no relevant documentation found)"
        return "\n\n---\n\n".join(
            f"Document {i + 1} (source: {source}):\n{text}"
            for i, (text, source) in enumerate(kept)
        )

    def generate(self, question):
        context = self.retrieve(question)
        messages = self.prompt.format_messages(context=context, question=question)

        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": messages[0].content},
                {"role": "user", "content": messages[1].content},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    rag = RAGSystem()
    test_query = "What is a bucket policy in Amazon S3?"
    print(f"Question: {test_query}")
    print(rag.generate(test_query))
