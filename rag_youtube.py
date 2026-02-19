"""
YouTube RAG: fetch transcript by video ID, chunk, embed with OpenAI, store in FAISS,
then answer CLI queries using a context-augmented prompt.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


def fetch_transcript(video_id: str) -> str:
    """Fetch YouTube transcript by video ID and return concatenated text."""
    try:
        api = YouTubeTranscriptApi()
        result = api.fetch(video_id)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch transcript for video ID '{video_id}'. "
            "Check the ID, captions availability, or try again later."
        ) from e
    if result is None or not result.snippets:
        raise RuntimeError(f"No transcript found for video ID '{video_id}'.")
    return " ".join(s.text for s in result.snippets)


FAISS_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_stores")


def build_rag(video_id: str):
    """Load existing FAISS store for the video, or build one from scratch."""
    model = os.environ.get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=model)

    store_path = os.path.join(FAISS_STORE_DIR, video_id)
    if os.path.isdir(store_path):
        print("Loading existing vector store from disk...")
        vector_store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        transcript = fetch_transcript(video_id)
        doc = Document(page_content=transcript)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([doc])

        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(store_path, exist_ok=True)
        vector_store.save_local(store_path)
        print("Vector store saved to disk.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever


def answer_query(retriever, question: str, llm) -> None:
    """Retrieve context, augment prompt, and stream the LLM answer."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    template = (
        "Use the following context from a YouTube video transcript to answer the question. "
        "If you cannot answer from the context, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate.from_template(template)
    formatted = prompt.format(context=context, question=question)

    print("\nAnswer: ", end="", flush=True)
    for chunk in llm.stream(formatted):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        if content:
            print(content, end="", flush=True)
    print()


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in a .env file (or in your environment).")
        return

    video_id = input("Enter YouTube video ID: ").strip()
    if not video_id:
        print("Video ID is required.")
        return

    print("Fetching transcript and building vector store...")
    retriever = build_rag(video_id)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    print("Ready. Ask questions about the video (or 'quit' to exit).")
    while True:
        question = input("\nYour question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        print("Thinking...")
        try:
            answer_query(retriever, question, llm)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
