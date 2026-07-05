from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

loader=PyPDFLoader("tucalender.pdf")
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
    collection_name="tu_calendar_db"
)

retriever=vectorstore.as_retriever(
    search_kwargs={"k":10}
)

prompt=PromptTemplate(
template="""
You are an assistant for TU Academic Calendar.
Answer ONLY using the given context.
If information is not found, say "Not mentioned in TU calendar".

Context:
{context}

Question:
{query}

Answer:
""",
input_variables=["context","query"]
)

chain=prompt|llm

print("===== TU Academic Calendar Assistant =====")
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