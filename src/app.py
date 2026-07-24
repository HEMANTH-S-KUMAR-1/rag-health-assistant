"""
app.py
------
Step 7: Flask web UI for the RAG Q&A assistant.
A clean, modern single-page interface to ask questions and see answers
with source citations.

Usage:
    python src/app.py
    -> opens at http://127.0.0.1:5000
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Ensure src/ is on the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from retrieve import Retriever
from generate import call_llm
from verify import verify_citations

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

# Lazy-load retriever (only once)
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        retriever = get_retriever()
        results = retriever.search_multihop(question, top_k=3)
        answer = call_llm(question, results)
        citations = verify_citations(answer, results)

        sources = []
        for r in results:
            sources.append({
                "title": r["chunk"]["title"],
                "score": round(r["score"], 3),
                "snippet": r["chunk"]["text"][:200].replace("\n", " "),
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
            "citations": citations,
        })

    except (EnvironmentError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting RAG Q&A web UI...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, port=5000)
