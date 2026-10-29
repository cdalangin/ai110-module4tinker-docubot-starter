"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re
import string

# Common words ignored when scoring, so filler words don't count as matches
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "do", "does", "i", "how",
              "what", "which", "where", "when", "why", "to", "of", "in", "on",
              "for", "and", "or", "it", "my", "can", "we", "you", "there"}

# Refusal guardrail: sections scoring below this are not returned
MIN_SCORE = 2

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            for raw in text.lower().split():
                token = raw.strip(string.punctuation)
                if token:
                    index.setdefault(token, set()).add(filename)
        return {token: sorted(files) for token, files in index.items()}

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_words = {w.strip(string.punctuation) for w in query.lower().split()}
        query_words -= STOP_WORDS | {""}
        text_words = {w.strip(string.punctuation) for w in text.lower().split()}
        return sum(1 for word in query_words if word in text_words)

    def split_sections(self, text, max_chars=800):
        """
        Split a document into sections, one per Markdown heading.
        A section that is longer than max_chars is split further by paragraph
        (blank lines), so results stay small even without headings.
        """
        sections = []
        for block in re.split(r"(?m)^(?=#{1,6} )", text):
            block = block.strip()
            if not block:
                continue
            if len(block) <= max_chars:
                sections.append(block)
                continue

            # Too long: pack paragraphs together up to max_chars
            current = ""
            for paragraph in block.split("\n\n"):
                if current and len(current) + len(paragraph) > max_chars:
                    sections.append(current.strip())
                    current = ""
                current += paragraph + "\n\n"
            if current.strip():
                sections.append(current.strip())
        return sections

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        query_words = {w.strip(string.punctuation) for w in query.lower().split()}
        query_words -= STOP_WORDS | {""}

        # Use the index to find only documents sharing at least one word with the query
        candidates = set()
        for word in query_words:
            candidates.update(self.index.get(word, []))

        # Score smaller sections instead of whole files
        scored = []
        for filename, text in self.documents:
            if filename in candidates:
                for section in self.split_sections(text):
                    score = self.score_document(query, section)
                    if score >= MIN_SCORE:
                        scored.append((score, filename, section))

        # Highest score first; filename breaks ties so order is stable
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(filename, text) for _, filename, text in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
