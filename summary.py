"""
College Assistant Agent (simplified)
No database — just asks the student directly whether they were absent.
If they were, it reads the class notes from a file (.pdf or .txt) and
has the LLM summarize them.

Requires: pypdf (for PDF text extraction)
    pip install pypdf
"""

from unittest import loader

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

from langchain_community.document_loaders import PyPDFLoader
PdfReader=PyPDFLoader("OS.pdf")
document=loader.load()

from langchain_community.document_loaders import PyPDFLoader
PdfReader=PyPDFLoader("TOC.pdf")
document=loader.load()

from langchain_community.document_loaders import PyPDFLoader
PdfReader=PyPDFLoader("Ai.pdf")
document=loader.load()

@tool
def read_notes_file(file_path: str) -> str:
    """Read the class notes from a file on disk and return the raw text.
    Supports .pdf and .txt files. file_path should be the full path to
    the notes file."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip() or "No extractable text found in this PDF."

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    return f"Unsupported file type: {ext}. Please provide a .pdf or .txt file."


tools = [read_notes_file]
llm_with_tools = llm.bind_tools(tools)
TOOL_MAP = {t.name: t for t in tools}


def run_agent(query: str) -> str:
    messages = [("human", query)]

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
    was_absent = input("Were you absent from class today? (yes/no): ").strip().lower()

    if was_absent in ("yes", "y"):
        file_path = input("Enter the path to the notes file (.pdf or .txt): ").strip()

        query = (
            f"Read the notes file at '{file_path}' and give me a clear 5-point "
            f"bullet summary of what was covered in class, written so a student "
            f"who missed the lecture can catch up quickly."
        )
        result = run_agent(query)
        print("\n--- Summary of missed class ---\n")
        print(result)

    else:
        print("Great, no notes to catch up on. Attendance marked present.")