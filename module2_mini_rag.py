"""
Module 2: Mini RAG System (No External Files)
==============================================
A Retrieval-Augmented Generation system for internal operational
question answering with source citation and hallucination prevention.

Knowledge Base:
- DOC_1: Support hours are Monday to Friday from 8am to 6pm.
- DOC_2: Quotations are answered within a maximum of 24 business hours.
- DOC_3: Contracts require legal approval before being signed.
"""

from typing import Dict, List, Tuple
import re
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    content: str
    keywords: List[str]


class MiniRAGSystem:
    """
    A lightweight RAG implementation using keyword matching and semantic
    similarity heuristics. In production, this would use vector embeddings
    and a proper vector database.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {
            "DOC_1": Document(
                doc_id="DOC_1",
                content="Support hours are Monday to Friday from 8am to 6pm.",
                keywords=[
                    "support",
                    "hours",
                    "monday",
                    "friday",
                    "8am",
                    "6pm",
                    "schedule",
                    "time",
                    "open",
                ],
            ),
            "DOC_2": Document(
                doc_id="DOC_2",
                content="Quotations are answered within a maximum of 24 business hours.",
                keywords=[
                    "quotation",
                    "quote",
                    "answer",
                    "respond",
                    "24",
                    "hours",
                    "business",
                    "time",
                    "response",
                ],
            ),
            "DOC_3": Document(
                doc_id="DOC_3",
                content="Contracts require legal approval before being signed.",
                keywords=[
                    "contract",
                    "legal",
                    "approval",
                    "sign",
                    "agreement",
                    "legal",
                    "review",
                ],
            ),
        }

        self.semantic_expansions = {
            "support": ["help", "assistance", "customer service"],
            "quotation": ["quote", "price", "cost", "estimate"],
            "contract": ["agreement", "deal", "sign", "legal document"],
            "hours": ["time", "schedule", "when", "available"],
        }

    def calculate_relevance_score(self, question: str, doc: Document) -> float:
        """
        Calculates a relevance score using multiple heuristics:
        1. Direct keyword matching (30% weight)
        2. Semantic expansion matching (20% weight)
        3. Question-type detection (40% weight)
        """
        question_lower = question.lower()
        question_words = set(re.findall(r"\w+", question_lower))

        score = 0.0

        keyword_matches = len(set(doc.keywords) & question_words)
        score += keyword_matches * 0.3

        for keyword, expansions in self.semantic_expansions.items():
            if keyword in doc.keywords:
                for expansion in expansions:
                    if expansion in question_lower:
                        score += 0.2

        if "when" in question_lower and "hours" in doc.content.lower():
            score += 0.4
        if "how long" in question_lower and "hours" in doc.content.lower():
            score += 0.4
        if (
            "can" in question_lower or "need" in question_lower
        ) and "require" in doc.content.lower():
            score += 0.3

        return score

    def retrieve_documents(
        self, question: str, threshold: float = 0.1
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves and ranks documents by relevance score.
        Returns documents above threshold, sorted by score.
        """
        scored_docs = []

        for doc_id, doc in self.documents.items():
            score = self.calculate_relevance_score(question, doc)
            if score >= threshold:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    def generate_answer(
        self, question: str, relevant_docs: List[Tuple[Document, float]]
    ) -> str:
        """
        Generates an answer strictly from retrieved documents.
        If no documents are relevant, returns a fallback message.
        """
        if not relevant_docs:
            return "I don't have information available to answer this question."

        doc_contents = [doc.content for doc, score in relevant_docs]

        if len(relevant_docs) == 1:
            answer = f"Based on our records, {relevant_docs[0][0].content.lower()}"
        else:
            answer_parts = [doc.content for doc, _ in relevant_docs]
            answer = "According to our documentation: " + " Additionally, ".join(
                answer_parts
            )

        return answer

    def answer_question(self, question: str) -> Dict:
        """
        Main function: receives question, retrieves docs, generates answer.

        Args:
            question: User's natural language question

        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        relevant_docs = self.retrieve_documents(question)

        answer = self.generate_answer(question, relevant_docs)

        sources = [doc.doc_id for doc, score in relevant_docs]

        return {"answer": answer, "sources": sources}


_rag_system = MiniRAGSystem()


def answer_question(question: str) -> Dict:
    """
    Standalone interface for the RAG system.
    """
    return _rag_system.answer_question(question)


def demonstrate_document_relevance():
    """Demonstrates how document relevance is determined."""

    print("=" * 70)
    print("MODULE 2 - TASK 1: DOCUMENT RELEVANCE DETERMINATION")
    print("=" * 70)

    rag = MiniRAGSystem()

    test_questions = [
        "When is support available?",
        "How long does it take to get a quote?",
        "Can I sign a contract without approval?",
        "What's the company policy on remote work?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")

        scored_docs = rag.retrieve_documents(q)

        if scored_docs:
            for doc, score in scored_docs:
                print(f'  {doc.doc_id}: score={score:.2f}, content="{doc.content}"')

            result = rag.answer_question(q)
            print(f"  Answer: {result['answer']}")
            print(f"  Sources: {result['sources']}")
        else:
            print("  No relevant documents found")
            result = rag.answer_question(q)
            print(f"  Answer: {result['answer']}")


def explain_hallucination_prevention():
    """Explains the hallucination prevention strategy."""

    print("\n" + "=" * 70)
    print("MODULE 2 - TASK 1: HALLUCINATION PREVENTION STRATEGY")
    print("=" * 70)

    strategies = [
        {
            "mechanism": "Answer Generation Constraint",
            "description": "The generate_answer() function only produces output derived from retrieved document content. No external knowledge is incorporated.",
        },
        {
            "mechanism": "Explicit Fallback",
            "description": "When no documents meet the relevance threshold, the system returns a clear 'I don't have information available' message rather than attempting to answer.",
        },
        {
            "mechanism": "Source Attribution",
            "description": "Every answer includes the source document IDs, enabling users and auditors to verify the information against original documents.",
        },
        {
            "mechanism": "No Extrapolation",
            "description": "The system does not synthesize new information or make inferences beyond what is explicitly stated in the source documents.",
        },
        {
            "mechanism": "Document-Only Content",
            "description": "The answer template uses the exact content from documents ('Based on our records...') rather than allowing creative reformulation that could introduce errors.",
        },
    ]

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['mechanism']}")
        print(f"   {strategy['description']}")


def scaling_strategy():
    """Describes the strategy for scaling to 100,000 documents."""

    print("\n" + "=" * 70)
    print("MODULE 2 - ADDITIONAL QUESTION: SCALING TO 100,000 DOCUMENTS")
    print("=" * 70)

    strategies = [
        {
            "title": "1. Vector Embeddings with Proper Vector Database",
            "description": "Replace keyword matching with dense vector embeddings (e.g., OpenAI text-embedding-3-small, or open-source alternatives like sentence-transformers). Store embeddings in a dedicated vector database (Pinecone, Weaviate, Milvus, or pgvector for PostgreSQL). This enables semantic similarity search at scale with O(log n) complexity using approximate nearest neighbor (ANN) algorithms like HNSW.",
        },
        {
            "title": "2. Chunking Strategy",
            "description": "Long documents must be split into semantically coherent chunks (typically 500-1000 tokens with overlap). Each chunk becomes a searchable unit, improving retrieval precision.",
        },
        {
            "title": "3. Retrieval Optimization",
            "description": "Hybrid Search: Combine vector similarity with BM25 lexical search for better recall. Reranking: Use a smaller, faster initial retrieval (top 100), then rerank with a cross-encoder model. Metadata Filtering: Enable pre-filtering by document type, date, department to narrow search space.",
        },
        {
            "title": "4. Caching and Index Optimization",
            "description": "Implement query result caching for frequent questions. Use approximate nearest neighbor (ANN) indexes with configurable precision-recall tradeoffs. Consider document clustering to narrow search space.",
        },
        {
            "title": "5. Pipeline Architecture",
            "description": "Question → Embedding → Vector Search → Reranking → Context Assembly → LLM Generation → Response",
        },
        {
            "title": "6. Cost Management",
            "description": "With 100K documents, embedding costs and query latency become significant. Consider: Smaller embedding models for initial retrieval, larger models for reranking. Quantized embeddings (8-bit vs 32-bit float). Batch embedding updates for new documents. Query result caching with TTL.",
        },
    ]

    for strategy in strategies:
        print(f"\n{strategy['title']}")
        print(f"   {strategy['description']}")


if __name__ == "__main__":
    demonstrate_document_relevance()
    explain_hallucination_prevention()
    scaling_strategy()
