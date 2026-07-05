#scans the constitution of nepal and give related answers to the user 

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pypdf import PdfReader
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"))

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

CONSTITUTION_FILE = "CON.pdf"
PERSIST_DIR = "./chroma_constitution_db"
COLLECTION_NAME = "constitution"


def build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"),
    )

    if os.path.exists(PERSIST_DIR):
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    if not os.path.exists(CONSTITUTION_FILE):
        raise FileNotFoundError(f"File not found: {CONSTITUTION_FILE}")

    reader = PdfReader(CONSTITUTION_FILE)
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": CONSTITUTION_FILE, "page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})



@tool
def retrieve_constitution(query: str) -> str:
    """Retrieve the passages from the Constitution of Nepal (CON.pdf) most
    relevant to the given query - e.g. a specific fine/penalty, an
    article about rights or government structure, or an amendment."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant passage found for that query."

    return "\n\n---\n\n".join(
        f"(page {doc.metadata.get('page', '?')}) {doc.page_content}" for doc in results
    )


tools = [retrieve_constitution]
llm_with_tools = llm.bind_tools(tools)
TOOL_MAP = {t.name: t for t in tools}

chatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful legal-reference assistant for the Constitution
        of Nepal. Use the retrieve_constitution tool with the user's
        question (or a good search phrase derived from it) to fetch the
        most relevant passages, then answer clearly and cite the
        article/part and page number the answer comes from. If the
        question is about a fine, penalty, or specific provision, quote
        or closely paraphrase the exact figure/wording found. If asked
        about a change or amendment, explain what changed and what its
        practical effect is, based only on what's in the retrieved text.
        If the retrieved passages don't actually answer the question,
        say so clearly instead of guessing or inventing legal content."""
    ),
    HumanMessagePromptTemplate.from_template("{question}"),
])


# Agent

def run_agent(messages: list) -> str:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = TOOL_MAP[tool_call["name"]]
            tool_result = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return response.content


if __name__ == "__main__":
    print("Ask me anything about the Constitution of Nepal (type 'exit' to quit).")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        messages = chatPrompt.format_messages(question=question)
        result = run_agent(messages)
        print("\n--- Answer ---\n")
        print(result)