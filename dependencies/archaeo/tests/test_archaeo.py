"""Tests for archaeo code archaeology engine."""

from __future__ import annotations

import pytest
from datetime import datetime

from archaeo import (
    ArchaeoConfig,
    ArchaeoEngine,
    AuthorProfile,
    ChangeType,
    CodeRegion,
    FileHistory,
    KnowledgeLevel,
)
from archaeo.ryzanstein import RyzansteinArchaeoClient


class TestChangeType:
    def test_all_variants(self) -> None:
        assert len(ChangeType) == 6

    def test_values(self) -> None:
        assert ChangeType.REFACTOR.value == "refactor"


class TestKnowledgeLevel:
    def test_all_variants(self) -> None:
        assert len(KnowledgeLevel) == 5

    def test_ordering(self) -> None:
        assert KnowledgeLevel.CREATOR.value == "creator"
        assert KnowledgeLevel.UNKNOWN.value == "unknown"


class TestFileHistory:
    def test_defaults(self) -> None:
        fh = FileHistory(path="test.py")
        assert fh.total_commits == 0
        assert fh.authors == []
        assert fh.churn_score == 0.0

    def test_with_data(self) -> None:
        fh = FileHistory(
            path="src/main.rs",
            total_commits=42,
            authors=["alice", "bob"],
            churn_score=150.0,
        )
        assert fh.total_commits == 42
        assert len(fh.authors) == 2


class TestAuthorProfile:
    def test_defaults(self) -> None:
        ap = AuthorProfile(name="alice", email="alice@example.com")
        assert ap.files_touched == 0
        assert ap.total_commits == 0

    def test_knowledge_areas(self) -> None:
        ap = AuthorProfile(
            name="bob",
            email="bob@example.com",
            knowledge_areas={"src/lib.rs": KnowledgeLevel.EXPERT},
        )
        assert ap.knowledge_areas["src/lib.rs"] == KnowledgeLevel.EXPERT


class TestCodeRegion:
    def test_defaults(self) -> None:
        cr = CodeRegion(file="test.py", start_line=1, end_line=10)
        assert cr.knowledge_level == KnowledgeLevel.UNKNOWN
        assert cr.owner == ""


class TestArchaeoConfig:
    def test_defaults(self) -> None:
        cfg = ArchaeoConfig()
        assert cfg.ryzanstein_url == "http://localhost:8000"
        assert cfg.max_history_depth == 1000
        assert cfg.churn_window_days == 90

    def test_custom(self) -> None:
        cfg = ArchaeoConfig(max_history_depth=50)
        assert cfg.max_history_depth == 50


class TestArchaeoEngine:
    def test_init_default(self) -> None:
        engine = ArchaeoEngine()
        assert engine.config.max_history_depth == 1000

    def test_analyze_file_nonexistent_repo(self) -> None:
        engine = ArchaeoEngine()
        history = engine.analyze_file("/nonexistent", "foo.py")
        assert history.path == "foo.py"
        assert history.total_commits == 0

    def test_analyze_author_nonexistent_repo(self) -> None:
        engine = ArchaeoEngine()
        profile = engine.analyze_author("/nonexistent", "alice")
        assert profile.name == "alice"
        assert profile.total_commits == 0

    def test_find_experts_nonexistent_repo(self) -> None:
        engine = ArchaeoEngine()
        experts = engine.find_experts("/nonexistent", "foo.py")
        assert experts == []

    def test_hotspots_nonexistent_repo(self) -> None:
        engine = ArchaeoEngine()
        spots = engine.hotspots("/nonexistent")
        assert spots == []


class TestRyzansteinArchaeoClient:
    @pytest.mark.asyncio
    async def test_health_check_offline(self) -> None:
        client = RyzansteinArchaeoClient()
        assert await client.health_check() is False

    def test_fallback_summary(self) -> None:
        client = RyzansteinArchaeoClient()
        result = client._fallback_summary("test.py", ["abc123", "def456"])
        assert "test.py" in result
        assert "2 commits" in result

    def test_fallback_summary_empty(self) -> None:
        client = RyzansteinArchaeoClient()
        result = client._fallback_summary("test.py", [])
        assert "0 commits" in result
