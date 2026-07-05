"""
College Assistant Agent (simplified)
-------------------------------------
No database — just asks the student directly whether they were absent,
and if so, which lesson/topic they missed. It then reads OS.pdf and has
the LLM summarize the relevant section.

Requires: langchain_community + pypdf (PyPDFLoader depends on pypdf under the hood)
    pip install langchain_community pypdf
"""

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model_name="llama-3.3-70b-versatile")

NOTES_FILE = "OS.pdf"



@tool
def read_notes_file() -> str:
    """Read the class notes from OS.pdf and return the raw text."""
    if not os.path.exists(NOTES_FILE):
        return f"File not found: {NOTES_FILE}"

    loader = PyPDFLoader(NOTES_FILE)
    document = loader.load()
    text = "\n".join(page.page_content for page in document)
    return text.strip() or "No extractable text found in this PDF."


tools = [read_notes_file]
llm_with_tools = llm.bind_tools(tools)
TOOL_MAP = {t.name: t for t in tools}



chatPrompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful college assistant. A student missed class and "
        "needs to catch up on a specific lesson/topic. Use the "
        "read_notes_file tool to fetch the full contents of OS.pdf, then "
        "find the section relevant to the topic the student mentions and "
        "summarize ONLY that part in exactly 5 clear bullet points so the "
        "student can quickly understand what they missed. If the topic "
        "isn't found in the notes, say so clearly instead of guessing."
    ),
    HumanMessagePromptTemplate.from_template(
        "I was absent from class and missed the lesson on '{topic}'. "
        "Please summarize that part of OS.pdf for me."
    ),
])


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
    was_absent = input("Were you absent from class today? (yes/no): ").strip().lower()

    if was_absent in ("yes", "y"):
        topic = input("Which lesson or topic did you miss? ").strip()

        messages = chatPrompt.format_messages(topic=topic)
        result = run_agent(messages)

        print("\n--- Summary of missed lesson ---\n")
        print(result)

    else:
        print("Great, no notes to catch up on. Attendance marked present.")