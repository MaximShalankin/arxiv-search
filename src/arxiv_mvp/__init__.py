"""arxiv-search: hybrid retrieval and RAG over an arXiv corpus."""
__version__ = "0.1.0"

# Optional .env loading: keeps local runs convenient without changing the
# process environment when variables are already set.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a declared dependency
    pass
