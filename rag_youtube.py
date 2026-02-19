"""
RAG tool: fetch content from YouTube transcripts or PDF files, chunk, embed with
OpenAI, store in FAISS, then answer CLI queries using a context-augmented prompt.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader
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


def load_pdf_pages(pdf_path: str, pages: list[int] | None = None) -> list[Document]:
    """Extract text from selected PDF pages and return as Documents.

    Args:
        pdf_path: Path to the PDF file.
        pages: 0-indexed page numbers to extract. None means all pages.
    """
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    if pages is None:
        pages = list(range(total))
    docs = []
    for p in pages:
        if p < 0 or p >= total:
            print(f"  Skipping page {p + 1} (out of range, PDF has {total} pages)")
            continue
        text = reader.pages[p].extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": p + 1}))
    if not docs:
        raise RuntimeError("No text could be extracted from the selected pages.")
    return docs


FAISS_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_stores")


def build_rag(store_key: str, documents: list[Document] | None = None):
    """Load existing FAISS store, or build one from the provided documents.

    Args:
        store_key: Unique identifier used as the directory name for persistence.
        documents: Pre-built Document list (only needed when creating a new store).
    """
    model = os.environ.get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=model)

    store_path = os.path.join(FAISS_STORE_DIR, store_key)
    if os.path.isdir(store_path):
        print("Loading existing vector store from disk...")
        vector_store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        if not documents:
            raise RuntimeError("No documents provided and no cached store found.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

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
        "Use the following context from a document to answer the question. "
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


def parse_page_selection(selection: str, total_pages: int) -> list[int] | None:
    """Parse user page input into a list of 0-indexed page numbers.

    Accepts: 'all', a single number like '5', or a range like '10-50'.
    Returns None for 'all'.
    """
    selection = selection.strip().lower()
    if selection == "all":
        return None
    if "-" in selection:
        parts = selection.split("-", 1)
        start, end = int(parts[0]), int(parts[1])
        if start < 1 or end < start or end > total_pages:
            raise ValueError(
                f"Invalid range. Must be between 1 and {total_pages}."
            )
        return list(range(start - 1, end))  # convert to 0-indexed
    page = int(selection)
    if page < 1 or page > total_pages:
        raise ValueError(f"Page must be between 1 and {total_pages}.")
    return [page - 1]  # convert to 0-indexed


def run_youtube_flow() -> str | None:
    """Prompt for video ID, fetch transcript, and return (store_key, documents)."""
    video_id = input("Enter YouTube video ID: ").strip()
    if not video_id:
        print("Video ID is required.")
        return None
    transcript = fetch_transcript(video_id)
    docs = [Document(page_content=transcript)]
    return video_id, docs


def run_pdf_flow() -> tuple[str, list[Document]] | None:
    """Prompt for PDF path and page selection, return (store_key, documents)."""
    pdf_path = input("Enter path to PDF file: ").strip()
    if not pdf_path or not os.path.isfile(pdf_path):
        print("Valid PDF file path is required.")
        return None

    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    print(f"PDF has {total} pages.")

    selection = input(
        "Pages to embed — single page (e.g. 5), range (e.g. 10-50), or 'all': "
    ).strip()
    try:
        pages = parse_page_selection(selection, total)
    except ValueError as e:
        print(f"Invalid selection: {e}")
        return None

    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    if pages is None:
        store_key = f"pdf_{filename}_all"
    elif len(pages) == 1:
        store_key = f"pdf_{filename}_p{pages[0] + 1}"
    else:
        store_key = f"pdf_{filename}_p{pages[0] + 1}-{pages[-1] + 1}"

    docs = load_pdf_pages(pdf_path, pages)
    print(f"Extracted text from {len(docs)} page(s).")
    return store_key, docs


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in a .env file (or in your environment).")
        return

    print("Source type:\n  1) YouTube video\n  2) PDF file")
    choice = input("Choose (1 or 2): ").strip()

    if choice == "1":
        result = run_youtube_flow()
    elif choice == "2":
        result = run_pdf_flow()
    else:
        print("Invalid choice.")
        return

    if result is None:
        return
    store_key, docs = result

    print("Building vector store...")
    retriever = build_rag(store_key, docs)
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    print("Ready. Ask questions about the document (or 'quit' to exit).")
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
