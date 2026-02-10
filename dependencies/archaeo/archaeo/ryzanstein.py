"""Ryzanstein integration for archaeo."""

from __future__ import annotations

import httpx

from archaeo import ArchaeoConfig


class RyzansteinArchaeoClient:
    """Communicates with Ryzanstein for AI-enhanced code archaeology."""

    def __init__(self, config: ArchaeoConfig | None = None) -> None:
        self.config = config or ArchaeoConfig()

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.config.ryzanstein_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def summarize_history(self, file_path: str, commits: list[str]) -> str:
        """Ask Ryzanstein to summarize the evolution of a file."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.config.ryzanstein_url}/v1/chat/completions",
                    json={
                        "model": "ryzanstein",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Summarize code evolution from commit messages.",
                            },
                            {
                                "role": "user",
                                "content": f"File: {file_path}\nCommits: {', '.join(commits[:20])}",
                            },
                        ],
                    },
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            return self._fallback_summary(file_path, commits)

    def _fallback_summary(self, file_path: str, commits: list[str]) -> str:
        n = len(commits)
        return f"{file_path}: {n} commits recorded. Most recent: {commits[0] if commits else 'none'}"
