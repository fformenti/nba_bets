"""What the LLM completes: the cue that ends the prompt and the answer after it.

Two modes share every line of the prompt before the cue:

- ``regression``: ``Home margin =`` then the signed, era-scaled margin (``+7``).
- ``classification``: ``Home team`` then `` won`` / `` lost``.

Both are scored the same way. ``P(home)`` is the two-way softmax over the
first completion token - the sign, or the word - so the two modes land in one
``p_home`` column, are Platt-recalibrated alike and compare on the same games.
Everything that differs between them lives in a :class:`TargetSpec`; the rest
of the pipeline asks the spec instead of branching on the mode.

The leading space of `` won`` is not a filler token: Llama's BPE folds it into
the word (`` won`` is one token), and "Home team won" is text the base model
has a prior over, where "Home teamwon" is not.
"""

from dataclasses import dataclass
from typing import Literal

TargetMode = Literal["regression", "classification"]

# A classification dataset's Hub name ends in this, and a regression one's never
# does, so pushing one mode can never overwrite the other's dataset. Regression
# keeps its original unsuffixed names (nba-bets, nba-bets-compact, ...), which
# the existing runs were trained on. LLMDataConfig enforces it at config load.
CLASSIFICATION_DATASET_SUFFIX = "-classification"


@dataclass(frozen=True)
class TargetSpec:
    cue: str  # the prompt's last line
    home_token: str  # first completion token meaning "home won"
    away_token: str
    has_magnitude: bool  # a margin token follows the label

    def completion(self, margin: float, scale: float = 1.0) -> str:
        """``+7`` / ``-12`` in reference-era points, or `` won`` / `` lost``.

        ``scale`` is the per-game ``margin_scale``; dividing by it is the era
        scaling (1.0 = raw points). It is positive, so it never changes the
        winner, and the NBA has no ties.
        """
        if not self.has_magnitude:
            return self.home_token if margin > 0 else self.away_token
        # int(): under numpy < 2, round() of a numpy float stays a float ("+7.0").
        value = int(round(margin / scale))  # noqa: RUF046
        return f"{self.home_token if value >= 0 else self.away_token}{abs(value)}"

    def parse(self, raw) -> float:
        """A greedy answer as a signed number: the margin, or ±1 for won/lost, 0 if unreadable."""
        if self.has_magnitude:
            from src.ml.llm.testers import RegressionTester

            return RegressionTester.post_process(raw)
        words = str(raw).split(maxsplit=1)
        answers = {self.home_token.strip(): 1.0, self.away_token.strip(): -1.0}
        return answers.get(words[0], 0.0) if words else 0.0


SPECS: dict[str, TargetSpec] = {
    "regression": TargetSpec("Home margin =", "+", "-", has_magnitude=True),
    "classification": TargetSpec("Home team", " won", " lost", has_magnitude=False),
}


def spec_of_completion(completion: str) -> TargetSpec:
    """The mode a stored completion was built in, so a saved CSV needs no config."""
    outcome = SPECS["classification"]
    words = {outcome.home_token.strip(), outcome.away_token.strip()}
    return outcome if str(completion).strip() in words else SPECS["regression"]


def check_dataset_matches(row: dict, spec: TargetSpec, prompt_field: str = "prompt") -> None:
    """Fail at load if the config's ``target_mode`` and the dataset disagree.

    The mode decides both the cue and the completion, and they come from the
    Hub dataset, not the config - so a config pointed at the other mode's
    dataset would otherwise train for 20 minutes on the wrong task.
    """
    if not row[prompt_field].endswith(spec.cue) or spec_of_completion(row["completion"]) is not spec:
        raise ValueError(
            f"data.target_mode expects prompts ending {spec.cue!r}, but the dataset's "
            f"first row ends {row[prompt_field][-20:]!r} -> {row['completion']!r}. "
            "Check data.dataset_name against data.target_mode."
        )
