# archaeo

Code archaeology and history analysis engine for the Ryzanstein LLM ecosystem.

## Overview

archaeo analyzes git history to extract knowledge patterns, expertise maps, code hotspots, and temporal evolution data. Used by Ryzanstein for context-aware code understanding and intelligent code review.

## Features

- **File History Analysis** — commit count, authors, change frequency, churn score
- **Author Profiling** — expertise level, files touched, lines added/removed
- **Expert Detection** — identify code owners by commit frequency
- **Hotspot Detection** — find most frequently changed files
- **Ryzanstein Integration** — AI-powered commit history summarization

## Quick Start

```python
from archaeo import ArchaeoEngine

engine = ArchaeoEngine()
history = engine.analyze_file("/path/to/repo", "src/main.py")
print(f"Commits: {history.total_commits}, Authors: {history.authors}")

experts = engine.find_experts("/path/to/repo", "src/main.py")
for name, level in experts:
    print(f"  {name}: {level.value}")
```

## License

AGPL-3.0
