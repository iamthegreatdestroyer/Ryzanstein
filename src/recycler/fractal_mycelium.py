import numpy as np
from typing import List, Dict
import random

class FractalMycelium:
    def __init__(self, branch_factor: int = 8, depth: int = 6):
        self.branch_factor = branch_factor
        self.depth = depth
        self.archive = []  # novelty archive

    def grow(self, prompt_tokens: List[int], temperature: float = 0.9) -> List[int]:
        mycelium = [prompt_tokens]
        for level in range(self.depth):
            new_branches = []
            for branch in mycelium[-self.branch_factor:]:
                for _ in range(self.branch_factor):
                    mutated = branch.copy()
                    # fractal mutation (non-linear token exploration)
                    idx = random.randint(0, len(mutated)-1)
                    mutated[idx] = int(mutated[idx] * (1 + random.gauss(0, temperature)))
                    new_branches.append(mutated)
            mycelium.extend(new_branches)

        # Novelty collapse: keep only the most different path
        best = max(mycelium, key=lambda x: self._novelty_score(x))
        self.archive.append(best)
        return best[:len(prompt_tokens) + 256]  # limit output

    def _novelty_score(self, tokens: List[int]) -> float:
        if not self.archive:
            return 100.0
        distances = [np.mean(np.abs(np.array(tokens) - np.array(a))) for a in self.archive]
        return min(distances) * -1  # reward strangeness