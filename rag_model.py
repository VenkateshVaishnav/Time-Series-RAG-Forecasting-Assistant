from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 🔹 Knowledge (your insights)
knowledge = [
    "Sales are higher on weekends, especially Saturday and Sunday.",
    "Sales decrease mid-week, especially Wednesday and Thursday.",
    "December has peak sales due to holidays.",
    "Promotions increase sales significantly.",
]

# 🔹 Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 🔹 Vector DB
vector_db = FAISS.from_texts(knowledge, embeddings)
retriever = vector_db.as_retriever()

# 🔹 LLM
pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=100,
    pad_token_id=50256
)

llm = HuggingFacePipeline(pipeline=pipe)

# 🔹 RAG function
def rag_chain(query):
    docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a sales analyst.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return llm.invoke(prompt)