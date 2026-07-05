from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

loader=PyPDFLoader("B_Sc.pdf")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs=text_splitter.split_documents(documents)
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="my_data_records"
)

retriever=vectorstore.as_retriever(
    search_kwargs={"k":10}
)
prompt=PromptTemplate(
template="""
You are an AI assistant.

Answer the question using the context below.

If the answer is partially available, answer as completely as possible.

Context:
{context}

Question:
{query}

Answer:
""",
input_variables=["context","query"]
)

chain=prompt|llm

print("===== PDF Question Answering System =====")
print("Loading PDF...")
loader=PyPDFLoader("B_Sc.pdf")
documents=loader.load()

print("Creating chunks...")
docs=text_splitter.split_documents(documents)

print("Loading embedding model...")
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector database...")
vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="my_data_records"
)

print("Retriever ready.")
print("Type 'exit' to quit.\n")

while True:

    query=input("You: ")

    if query.lower()=="exit":
        break

    relevant_chunks=retriever.invoke(query)

    context="\n\n".join(
        doc.page_content for doc in relevant_chunks
    )

    response=chain.invoke({
        "context":context,
        "query":query
    })

    print("\nAssistant:")
    print(response.content)
    print()