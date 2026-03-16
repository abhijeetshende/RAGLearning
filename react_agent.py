"""
ReAct Agent with a Summarize Tool
Uses LangChain's create_agent (LangGraph-based ReAct loop) with the
hwchase17/react prompt pulled from LangChain Hub via LangSmith.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langsmith.client import Client
from langchain_community.tools import DuckDuckGoSearchResults



@tool
def summarize_text(text: str) -> str:
    """Summarize a given text paragraph into 5-6 concise bullet points capturing the main ideas."""
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    prompt = (
        "You are a summarization assistant. "
        "Read the following text and produce exactly 5-6 bullet points "
        "that capture the main ideas. Each bullet should be a concise sentence.\n\n"
        f"Text:\n{text}\n\n"
        "Bullet-point summary:"
    )

    response = llm.invoke(prompt)
    return response.content


@tool
def answer_question(question: str) -> str:
    """Answer a general knowledge question. Use this for any factual, analytical, or informational query that is NOT a summarization request."""
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    prompt = (
        "Answer the following question accurately and concisely.\n\n"
        f"Question: {question}"
    )

    response = llm.invoke(prompt)
    return response.content

# @tool
# def search_web(query: str) -> str:
#     """Search the web for information on the given query."""
#     llm = ChatOpenAI(model="gpt-4.1", temperature=0)
#     prompt = (
#         "Search the web for information on the given query.\n\n"
#         f"Query: {query}"
#     )
#     response = llm.invoke(prompt)
#     return response.content

@tool
def search_web(query: str) -> str:
    """Search the web for information on the given query."""
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    prompt = (
        "Search the web for information on the given query.\n\n"
        f"Query: {query}"
    )
    search = DuckDuckGoSearchResults(max_results=4)
    search_results = search.run(query)
    prompt = (
        "Use the following search results to answer the question. "
        "If the results don't help, say so briefly.\n\n"
        f"Search results:\n{search_results}\n\n"
        f"Question: {query}"
    )
    for chunk in llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in a .env file (or in your environment).")
        return

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # Pull the ReAct prompt from LangChain Hub via LangSmith
    client = Client()
    react_prompt = client.pull_prompt("hwchase17/react")


    # Build tool descriptions and names for the ReAct prompt
    tools = [summarize_text, answer_question, search_web]
    tools_desc = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(t.name for t in tools)

    # Use the hwchase17/react prompt with actual tool info filled in
    system_prompt = react_prompt.template.format(
        tools=tools_desc,
        tool_names=tool_names,
        input="{user's question will appear in the message}",
        agent_scratchpad="",
    )

    agent = create_agent(llm, tools, system_prompt=system_prompt)

    print("ReAct Agent ready! Ask me anything or give me text to summarize.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        question = input("Your query: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print(f"\n>> Query: {question}\n")

        # Stream each step: tool calls, observations, final answer
        for step in agent.stream(
            {"messages": [("user", question)]},
            stream_mode="updates",
        ):

            for _, output in step.items():
                for msg in output.get("messages", []):
                    msg.pretty_print()
        print()


if __name__ == "__main__":
    main()
