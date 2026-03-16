"""
DuckDuck Go example
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in a .env file (or in your environment).")
        return
    search = DuckDuckGoSearchResults(max_results=4)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    print("Ask questions (empty or 'quit' to exit).\n")
    while True:
        question = input("Your question: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        search_results = search.run(question)
        prompt = (
            "Use the following search results to answer the question. "
            "If the results don't help, say so briefly.\n\n"
            f"Search results:\n{search_results}\n\n"
            f"Question: {question}"
        )
        for chunk in llm.stream(prompt):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    main()
