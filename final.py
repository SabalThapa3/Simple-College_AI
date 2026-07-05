from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

llm=ChatGroq(model_name="llama-3.3-70b-versatile")

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_retriever(pdf_path,collection_name):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    docs=text_splitter.split_documents(documents)
    vectorstore=Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=collection_name
    )
    return vectorstore.as_retriever(search_kwargs={"k":10})

bsc_retriever=create_retriever(
    "B_Sc.pdf",
    "bsc_database"
)

calendar_retriever=create_retriever(
    "tucalender.pdf",
    "calendar_database"
)

constitution_retriever=create_retriever(
    "cons.pdf",
    "constitution_database"
)

prompt=PromptTemplate(
template="""
You are an AI Assistant.Answer the user's question ONLY using the provided context. If the answer is partially available, answer as completely as possible. If the answer is not available in the context, reply: "I couldn't find this information in the selected document."
Context:
{context}

Question:
{query}

Answer:
""",
input_variables=["context","query"]
)

chain=prompt|llm

print("===== Multi PDF Question Answering System =====")

while True:
    print("\n1.B.Sc PDF")
    print("2.TU Academic Calendar")
    print("3.Constitution of Nepal")
    print("4.Search All PDFs")
    print("5.Exit")

    choice=input("\nEnter Choice: ")

    if choice=="5":
        break

    if choice not in ["1","2","3","4"]:
        print("Invalid Choice")
        continue

    query=input("You: ")

    if choice=="1":
        relevant_chunks=bsc_retriever.invoke(query)

    elif choice=="2":
        relevant_chunks=calendar_retriever.invoke(query)

    elif choice=="3":
        relevant_chunks=constitution_retriever.invoke(query)

    elif choice=="4":
        relevant_chunks=(
            bsc_retriever.invoke(query)+
            calendar_retriever.invoke(query)+
            constitution_retriever.invoke(query)
        )

    context="\n\n".join(
        doc.page_content for doc in relevant_chunks
    )

    response=chain.invoke({
        "context":context,
        "query":query
    })

    print("\nAssistant:")
    print(response.content)