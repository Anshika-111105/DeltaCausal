"""Unit tests for Grounded Causal Decision Assistant."""

import pytest
from src.assistant.knowledge_base import CausalKnowledgeBase
from src.assistant.rag_engine import DecisionAssistant


def test_knowledge_base_retrieval():
    kb = CausalKnowledgeBase()
    docs = kb.search_keyword("SUTVA unconfoundedness", top_k=2)
    assert len(docs) > 0
    assert "causal_assumptions_01" in [d.doc_id for d in docs]


def test_decision_assistant_grounded_answer():
    assistant = DecisionAssistant(artifacts_dir="artifacts")
    res = assistant.answer_query("What is the difference between response propensity and uplift targeting?")

    assert len(res.direct_answer) > 30
    assert "Sure Things" in res.direct_answer or "propensity" in res.direct_answer.lower()
    assert len(res.cited_sources) > 0
    assert any("doc_id" in c for c in res.cited_sources)
    assert len(res.causal_assumptions) >= 3

