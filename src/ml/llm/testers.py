import math
import re
from itertools import accumulate
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from src.ml.llm.target import SPECS, TargetSpec
from src.ml.plotting.llm import series_figures
from src.ml.results import build_predictions

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

DEFAULT_SIZE = 200


def ground_truth(datapoint) -> float:
    """The true point differential for one dataset row, in real points.

    Datasets built with era scaling carry the unscaled margin as ``margin``;
    the completion there is in reference-era points. Older datasets have only
    the completion, which then *is* the real margin — ``int`` parses the
    leading ``+`` natively.
    """
    if "margin" in datapoint:
        return float(datapoint["margin"])
    return float(int(datapoint["completion"]))


def margin_scale(datapoint) -> float:
    """Factor taking a completion-unit margin back to real points.

    ``margin_std / reference_std`` for the league spread over the game's
    trailing window — the same causal estimate the target was scaled by — and 1.0 for a
    dataset built unscaled. Positive, so it never changes a predicted winner.
    """
    return float(datapoint.get("margin_scale", 1.0))


class RegressionTester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []

    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value):
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            # The sign must cover both alternatives: in the old
            # ``[-+]?\d*\.\d+|\d+`` it bound to the decimal branch only, so
            # "-7" fell through to ``\d+`` and parsed as +7.
            match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", value)
            return float(match.group()) if match else 0
        else:
            return value

    def color_for(self, error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value) * margin_scale(datapoint)
        truth = ground_truth(datapoint)
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        title = f"game {datapoint['game_id']}"
        return title, guess, truth, error, color

    def chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        # Pre-format hover text
        df["hover"] = [
            f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        # Allow both axes to show negative values (the previous implementation
        # hard-clamped ranges to start at 0).
        min_val = float(min(df["truth"].min(), df["guess"].min()))
        max_val = float(max(df["truth"].max(), df["guess"].max()))
        if min_val == max_val:
            # Avoid Plotly throwing on a zero-span range.
            min_val -= 1
            max_val += 1

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            width=800,
            height=600,
        )

        # Assign customdata per trace (one color/category = one trace)
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Reference line y=x
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[min_val, max_val])
        fig.update_yaxes(range=[min_val, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self):
        n = len(self.errors)

        # Running mean and std (pure Python)
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # 95% confidence interval for mean
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        # Title with final stats
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Error: {final_mean:,.2f} ± {final_ci:,.2f}"

        # Plot
        fig = go.Figure()

        # Shaded confidence interval band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Main line with hover text showing CI
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Avg Error",
                customdata=list(
                    zip(
                        ci,
                    )
                ),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Error ($)",
            width=800,
            height=300,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        title = f"{self.title} results<br><b>Error:</b> ${average_error:,.2f} <b>MSE:</b> {mse:,.0f} <b>r²:</b> {r2:.1f}%"
        self.error_trend_chart()
        self.chart(title)

    def run(self):
        for i in tqdm(range(self.size)):
            title, guess, truth, error, color = self.run_datapoint(i)
            self.titles.append(title)
            self.guesses.append(guess)
            self.truths.append(truth)
            self.errors.append(error)
            self.colors.append(color)
            print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        clear_output(wait=True)
        self.report()


class ClassificationTester:
    """Evaluate a predictor as a binary classifier via sign agreement.

    A prediction is correct (1) when guess and truth share a sign
    (both > 0 or both <= 0), and incorrect (0) otherwise. ``spec`` parses the
    answer: a margin, or ±1 for `` won``/`` lost`` (src/ml/llm/target.py).
    """

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, spec: Optional[TargetSpec] = None):
        self.predictor = predictor
        self.data = data
        self.spec = spec or SPECS["regression"]
        self.title = title or RegressionTester.make_title(predictor)
        self.size = size
        self.titles = []
        self.game_ids = []
        self.guesses = []
        self.truths = []
        self.corrects = []
        self.colors = []

    @staticmethod
    def post_process(value):
        return RegressionTester.post_process(value)

    @staticmethod
    def is_correct(guess, truth) -> int:
        """1 if same sign (both positive or both non-positive), else 0."""
        return int((guess > 0) == (truth > 0))

    def color_for(self, correct: int) -> str:
        return "green" if correct else "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.spec.parse(value) * margin_scale(datapoint)
        truth = ground_truth(datapoint)
        correct = self.is_correct(guess, truth)
        color = self.color_for(correct)
        title = f"game {datapoint['game_id']}"
        return title, guess, truth, correct, color

    def predictions(self, run_id: str, split: str, model: Optional[str] = None) -> pd.DataFrame:
        """The scored games in the shared result schema (src/ml/results).

        Generation yields a margin, not a probability, so ``p_home`` is empty.
        ``model`` defaults to ``<run_id>-greedy``: ``inspect_llm`` writes the
        same adapter's first-token probabilities as ``<run_id>``, and the two
        must not overwrite each other in one predictions file.
        """
        return build_predictions(
            run_id=run_id,
            model=model or f"{run_id}-greedy",
            split=split,
            game_ids=self.game_ids,
            y_true=[int(t > 0) for t in self.truths],
            y_pred=[int(g > 0) for g in self.guesses],
            # A won/lost answer parses to ±1, which is no margin.
            pred_margin=self.guesses if self.spec.has_magnitude else None,
            true_margin=self.truths,
        )

    def report(self):
        """Show the figures in a notebook. Nothing is written to disk."""
        for fig in series_figures(self.predictions("notebook", "validation"), self.title).values():
            fig.show()

    def run(self):
        for i in tqdm(range(self.size)):
            title, guess, truth, correct, color = self.run_datapoint(i)
            self.titles.append(title)
            self.game_ids.append(self.data[i]["game_id"])
            self.guesses.append(guess)
            self.truths.append(truth)
            self.corrects.append(correct)
            self.colors.append(color)
            mark = "✓" if correct else "✗"
            print(f"{COLOR_MAP[color]}{mark} ", end="")
        clear_output(wait=True)
        self.report()


class BatchedClassificationTester(ClassificationTester):
    """:class:`ClassificationTester` that scores ``batch_size`` rows per call.

    ``batch_predictor(list[datapoint]) -> list[str]`` answers a whole chunk at
    once, so a GPU model decodes the batch in one ``generate`` instead of one
    prompt at a time. Grading is the parent's, unchanged. It draws nothing: the
    caller writes :meth:`predictions` and ``make plot`` draws them.
    """

    def __init__(
        self,
        batch_predictor,
        data,
        title=None,
        size=DEFAULT_SIZE,
        batch_size=16,
        spec: Optional[TargetSpec] = None,
    ):
        super().__init__(batch_predictor, data, title=title, size=size, spec=spec)
        self.batch_size = batch_size

    def run_batch(self, start: int) -> None:
        rows = [self.data[i] for i in range(start, min(start + self.batch_size, self.size))]
        values = self.predictor(rows)
        if len(values) != len(rows):
            raise ValueError(f"batch_predictor returned {len(values)} outputs for {len(rows)} rows")
        for datapoint, value in zip(rows, values):
            guess = self.spec.parse(value) * margin_scale(datapoint)
            truth = ground_truth(datapoint)
            correct = self.is_correct(guess, truth)
            self.titles.append(f"game {datapoint['game_id']}")
            self.game_ids.append(datapoint["game_id"])
            self.guesses.append(guess)
            self.truths.append(truth)
            self.corrects.append(correct)
            self.colors.append(self.color_for(correct))

    def metrics(self) -> dict:
        n_correct = sum(self.corrects)
        return {"accuracy": n_correct / self.size, "n_correct": n_correct, "size": self.size}

    def run(self) -> dict:
        self.size = min(self.size, len(self.data))
        with tqdm(total=self.size) as bar:
            for start in range(0, self.size, self.batch_size):
                self.run_batch(start)
                bar.update(len(self.corrects) - bar.n)
                bar.set_postfix(acc=f"{sum(self.corrects) / len(self.corrects):.1%}")
        return self.metrics()
