"""Telemetry dataclass for the Adaptive Candidate Search Engine — tracks
every decision the engine makes so the evidence panel can show *why* a
candidate was pruned, which stage caught it, and how many decode attempts
were saved.

Separate from search.py / adaptive_search.py so both engines and both
front ends (web + desktop) can import it without circular dependencies.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class CandidateTelemetry:
    """One entry per modulation candidate the adaptive engine evaluates."""

    modulation: str
    mod_rank: int = 0              # rank from feature classifier (1 = best)
    mod_score: float = 0.0         # cumulant-based plausibility (0–1)
    cfo_hz: float = 0.0            # estimated carrier offset applied
    sync_correlation: float = 0.0  # best sync-word correlation found
    sync_pruned: bool = False      # True → gatekeeper killed this branch
    fec_attempted: list[str] = field(default_factory=list)
    interleavers_attempted: list[str] = field(default_factory=list)
    decodes_run: int = 0           # how many FEC×interleaver combos tried
    early_exit: bool = False       # True → this candidate triggered early stop
    winner_fec: str | None = None
    winner_interleaver: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdaptiveSummary:
    """Aggregate telemetry for the whole adaptive search run."""

    total_candidates_evaluated: int = 0
    total_decodes_run: int = 0
    total_decodes_possible: int = 96  # 6 mod × 4 fec × 4 il
    modulations_pruned_by_classifier: list[str] = field(default_factory=list)
    modulations_pruned_by_sync: list[str] = field(default_factory=list)
    early_exit_triggered: bool = False
    search_space_reduction_pct: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["search_space_reduction_pct"] = round(self.search_space_reduction_pct, 1)
        return d
