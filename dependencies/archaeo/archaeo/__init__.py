"""
archaeo — Code archaeology and history analysis engine.

Tracks code evolution, identifies knowledge patterns, detects expertise
ownership, and provides temporal analysis of codebase changes.
Used by Ryzanstein for context-aware code understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChangeType(Enum):
    """Classification of code changes."""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    RENAME = "rename"
    MOVE = "move"
    REFACTOR = "refactor"


class KnowledgeLevel(Enum):
    """Author expertise level for a code region."""
    CREATOR = "creator"
    EXPERT = "expert"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    UNKNOWN = "unknown"


@dataclass
class FileHistory:
    """Historical record of a single file."""
    path: str
    created: datetime | None = None
    last_modified: datetime | None = None
    total_commits: int = 0
    authors: list[str] = field(default_factory=list)
    change_frequency: float = 0.0
    churn_score: float = 0.0


@dataclass
class AuthorProfile:
    """Developer expertise profile computed from commit history."""
    name: str
    email: str
    files_touched: int = 0
    total_commits: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    primary_languages: list[str] = field(default_factory=list)
    knowledge_areas: dict[str, KnowledgeLevel] = field(default_factory=dict)


@dataclass
class CodeRegion:
    """A specific region of code with ownership metadata."""
    file: str
    start_line: int
    end_line: int
    owner: str = ""
    knowledge_level: KnowledgeLevel = KnowledgeLevel.UNKNOWN
    last_author: str = ""
    last_modified: datetime | None = None


@dataclass
class ArchaeoConfig:
    """Configuration for archaeo engine."""
    ryzanstein_url: str = "http://localhost:8000"
    max_history_depth: int = 1000
    churn_window_days: int = 90
    min_commits_for_expert: int = 10
    ignore_patterns: list[str] = field(
        default_factory=lambda: [".git", "node_modules", "__pycache__", ".venv"]
    )


class ArchaeoEngine:
    """
    Main code archaeology engine. Analyzes git history to extract
    knowledge patterns, expertise maps, and temporal evolution data.
    """

    def __init__(self, config: ArchaeoConfig | None = None) -> None:
        self.config = config or ArchaeoConfig()
        self._file_cache: dict[str, FileHistory] = {}
        self._author_cache: dict[str, AuthorProfile] = {}

    def analyze_file(self, repo_path: str, file_path: str) -> FileHistory:
        """Analyze the history of a single file using git log."""
        try:
            import git
            repo = git.Repo(repo_path)
        except Exception:
            return FileHistory(path=file_path)

        commits = list(
            repo.iter_commits(paths=file_path, max_count=self.config.max_history_depth)
        )
        if not commits:
            return FileHistory(path=file_path)

        authors = list({c.author.name for c in commits if c.author})
        history = FileHistory(
            path=file_path,
            created=datetime.fromtimestamp(commits[-1].committed_date) if commits else None,
            last_modified=datetime.fromtimestamp(commits[0].committed_date) if commits else None,
            total_commits=len(commits),
            authors=authors,
            change_frequency=self._compute_frequency(commits),
            churn_score=self._compute_churn(commits, file_path),
        )
        self._file_cache[file_path] = history
        return history

    def analyze_author(self, repo_path: str, author_name: str) -> AuthorProfile:
        """Build expertise profile for a specific author."""
        try:
            import git
            repo = git.Repo(repo_path)
        except Exception:
            return AuthorProfile(name=author_name, email="")

        commits = [
            c for c in repo.iter_commits(max_count=self.config.max_history_depth)
            if c.author and c.author.name == author_name
        ]
        if not commits:
            return AuthorProfile(name=author_name, email="")

        files_touched: set[str] = set()
        lines_added = 0
        lines_removed = 0
        for c in commits:
            files_touched.update(c.stats.files.keys())
            lines_added += c.stats.total.get("insertions", 0)
            lines_removed += c.stats.total.get("deletions", 0)

        return AuthorProfile(
            name=author_name,
            email=commits[0].author.email if commits[0].author else "",
            files_touched=len(files_touched),
            total_commits=len(commits),
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

    def find_experts(self, repo_path: str, file_path: str) -> list[tuple[str, KnowledgeLevel]]:
        """Find the experts for a given file based on commit history."""
        history = self.analyze_file(repo_path, file_path)
        if not history.authors:
            return []

        # Simple heuristic: most commits → highest expertise
        try:
            import git
            repo = git.Repo(repo_path)
        except Exception:
            return [(a, KnowledgeLevel.UNKNOWN) for a in history.authors]

        author_commits: dict[str, int] = {}
        for c in repo.iter_commits(paths=file_path, max_count=self.config.max_history_depth):
            if c.author:
                author_commits[c.author.name] = author_commits.get(c.author.name, 0) + 1

        result: list[tuple[str, KnowledgeLevel]] = []
        sorted_authors = sorted(author_commits.items(), key=lambda x: x[1], reverse=True)
        for i, (name, count) in enumerate(sorted_authors):
            if i == 0:
                level = KnowledgeLevel.CREATOR
            elif count >= self.config.min_commits_for_expert:
                level = KnowledgeLevel.EXPERT
            elif count >= 3:
                level = KnowledgeLevel.CONTRIBUTOR
            else:
                level = KnowledgeLevel.REVIEWER
            result.append((name, level))
        return result

    def hotspots(self, repo_path: str, top_n: int = 10) -> list[FileHistory]:
        """Find the most frequently changed files (hotspots)."""
        try:
            import git
            repo = git.Repo(repo_path)
        except Exception:
            return []

        file_counts: dict[str, int] = {}
        for c in repo.iter_commits(max_count=self.config.max_history_depth):
            for f in c.stats.files:
                file_counts[f] = file_counts.get(f, 0) + 1

        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            FileHistory(path=f, total_commits=count, change_frequency=float(count))
            for f, count in sorted_files
        ]

    def _compute_frequency(self, commits: list[Any]) -> float:
        if len(commits) < 2:
            return 0.0
        first = datetime.fromtimestamp(commits[-1].committed_date)
        last = datetime.fromtimestamp(commits[0].committed_date)
        days = max((last - first).days, 1)
        return len(commits) / days

    def _compute_churn(self, commits: list[Any], file_path: str) -> float:
        total = 0
        for c in commits:
            stats = c.stats.files.get(file_path, {})
            total += stats.get("insertions", 0) + stats.get("deletions", 0)
        return float(total)
