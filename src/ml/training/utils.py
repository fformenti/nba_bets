import math
import re
from itertools import accumulate

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tqdm.auto import tqdm

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

DEFAULT_SIZE = 200


class Tester_Regressors:
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
        return (
            predictor.__name__.replace("__", ".")
            .replace("_", " ")
            .title()
            .replace("Gpt", "GPT")
        )

    @staticmethod
    def post_process(value):
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
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
        guess = self.post_process(value)
        truth = float(datapoint["point_diff"])
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        pieces = datapoint["text"].split("Title: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."
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
        ci = [
            1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)
        ]
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


class Tester_Classifiers:
    """Evaluate a predictor as a binary classifier via sign agreement.

    A prediction is correct (1) when guess and truth share a sign
    (both > 0 or both <= 0), and incorrect (0) otherwise.
    """

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        self.predictor = predictor
        self.data = data
        self.title = title or Tester_Regressors.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.corrects = []
        self.colors = []

    @staticmethod
    def post_process(value):
        return Tester_Regressors.post_process(value)

    @staticmethod
    def is_correct(guess, truth) -> int:
        """1 if same sign (both positive or both non-positive), else 0."""
        return int((guess > 0) == (truth > 0))

    def color_for(self, correct: int) -> str:
        return "green" if correct else "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = float(datapoint["point_diff"])
        correct = self.is_correct(guess, truth)
        color = self.color_for(correct)
        pieces = datapoint["text"].split("Title: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."
        return title, guess, truth, correct, color

    def chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "correct": self.corrects,
                "color": self.colors,
            }
        )

        df["hover"] = [
            f"{t}\nGuess={g:+.0f} Actual={y:+.0f}\n{'Correct' if c else 'Wrong'}"
            for t, g, y, c in zip(df["title"], df["guess"], df["truth"], df["correct"])
        ]

        min_val = float(min(df["truth"].min(), df["guess"].min()))
        max_val = float(max(df["truth"].max(), df["guess"].max()))
        if min_val == max_val:
            min_val -= 1
            max_val += 1

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "red": "red"},
            title=title,
            labels={"truth": "Actual Diff", "guess": "Predicted Diff"},
            width=800,
            height=600,
        )

        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Quadrant guides at zero (sign boundary)
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")

        fig.update_xaxes(range=[min_val, max_val])
        fig.update_yaxes(range=[min_val, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def accuracy_trend_chart(self):
        n = len(self.corrects)
        running_sums = list(accumulate(self.corrects))
        x = list(range(1, n + 1))
        running_acc = [s / i for s, i in zip(running_sums, x)]

        final_acc = running_acc[-1]
        title = f"{self.title} Accuracy: {final_acc:.1%}"

        fig = go.Figure()


class Tester_Classifiers:
    """Evaluate a predictor as a binary classifier via sign agreement.

    A prediction is correct (1) when guess and truth share a sign
    (both > 0 or both <= 0), and incorrect (0) otherwise.
    """

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        self.predictor = predictor
        self.data = data
        self.title = title or Tester_Regressors.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.corrects = []
        self.colors = []

    @staticmethod
    def post_process(value):
        return Tester_Regressors.post_process(value)

    @staticmethod
    def is_correct(guess, truth) -> int:
        """1 if same sign (both positive or both non-positive), else 0."""
        return int((guess > 0) == (truth > 0))

    def color_for(self, correct: int) -> str:
        return "green" if correct else "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = float(datapoint["point_diff"])
        correct = self.is_correct(guess, truth)
        color = self.color_for(correct)
        pieces = datapoint["text"].split("Title: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."
        return title, guess, truth, correct, color

    def chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "correct": self.corrects,
                "color": self.colors,
            }
        )

        df["hover"] = [
            f"{t}\nGuess={g:+.0f} Actual={y:+.0f}\n{'Correct' if c else 'Wrong'}"
            for t, g, y, c in zip(df["title"], df["guess"], df["truth"], df["correct"])
        ]

        min_val = float(min(df["truth"].min(), df["guess"].min()))
        max_val = float(max(df["truth"].max(), df["guess"].max()))
        if min_val == max_val:
            min_val -= 1
            max_val += 1

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "red": "red"},
            title=title,
            labels={"truth": "Actual Diff", "guess": "Predicted Diff"},
            width=800,
            height=600,
        )

        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Quadrant guides at zero (sign boundary)
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")

        fig.update_xaxes(range=[min_val, max_val])
        fig.update_yaxes(range=[min_val, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def accuracy_trend_chart(self):
        n = len(self.corrects)
        running_sums = list(accumulate(self.corrects))
        x = list(range(1, n + 1))
        running_acc = [s / i for s, i in zip(running_sums, x)]

        final_acc = running_acc[-1]
        title = f"{self.title} Accuracy: {final_acc:.1%}"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_acc,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Accuracy",
                hovertemplate="n=%{x}<br>Accuracy=%{y:.1%}<extra></extra>",
            )
        )
        fig.add_hline(
            y=0.5,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="chance (50%)",
        )

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Accuracy",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            width=800,
            height=300,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()

    def confidence_accuracy_chart(self):
        """Accuracy among top-k predictions by |predicted point_diff| (desc).

        Games with a larger predicted margin are treated as higher confidence;
        the curve shows whether those are easier to get right.
        """
        n = len(self.corrects)
        ranked = sorted(
            zip(self.guesses, self.corrects),
            key=lambda pair: abs(pair[0]),
            reverse=True,
        )
        margins = [abs(g) for g, _ in ranked]
        corrects_ranked = [c for _, c in ranked]

        running_sums = list(accumulate(corrects_ranked))
        x = [i / n for i in range(1, n + 1)]
        running_acc = [s / i for s, i in zip(running_sums, range(1, n + 1))]
        overall_acc = running_sums[-1] / n

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_acc,
                mode="lines",
                line=dict(width=3, color="steelblue"),
                name="Accuracy @ coverage",
                customdata=list(zip(range(1, n + 1), margins)),
                hovertemplate=(
                    "Coverage=%{x:.1%}<br>"
                    "Top-n=%{customdata[0]}<br>"
                    "|pred|=%{customdata[1]:.1f}<br>"
                    "Accuracy=%{y:.1%}<extra></extra>"
                ),
            )
        )
        fig.add_hline(
            y=overall_acc,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"overall ({overall_acc:.1%})",
        )
        fig.add_hline(
            y=0.5,
            line_width=1,
            line_dash="dot",
            line_color="lightgray",
            annotation_text="chance (50%)",
        )

        fig.update_layout(
            title=(
                f"{self.title} Confidence Curve<br>"
                f"<sup>Sorted by |predicted point_diff| descending</sup>"
            ),
            xaxis_title="Coverage (fraction of games, high |pred| first)",
            yaxis_title="Cumulative Accuracy",
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            width=800,
            height=400,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()

    def report(self):
        # Labels from sign: positive → 1, non-positive → 0
        y_true = [int(t > 0) for t in self.truths]
        y_pred = [int(g > 0) for g in self.guesses]
        acc = accuracy_score(y_true, y_pred)
        n_correct = sum(self.corrects)
        title = (
            f"{self.title} results<br>"
            f"<b>Accuracy:</b> {acc:.1%} "
            f"({n_correct}/{self.size})"
        )
        self.accuracy_trend_chart()
        self.confidence_accuracy_chart()
        self.chart(title)

    def run(self):
        for i in tqdm(range(self.size)):
            title, guess, truth, correct, color = self.run_datapoint(i)
            self.titles.append(title)
            self.guesses.append(guess)
            self.truths.append(truth)
            self.corrects.append(correct)
            self.colors.append(color)
            mark = "✓" if correct else "✗"
            print(f"{COLOR_MAP[color]}{mark} ", end="")
        clear_output(wait=True)
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE):
    Tester_Regressors(function, data, size=size).run()


def evaluate_classifier(function, data, size=DEFAULT_SIZE):
    Tester_Classifiers(function, data, size=size).run()
