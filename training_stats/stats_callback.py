"""Training stats: Score vs Episode (live plot + JSON/CSV/PNG at end)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

ROLLING_WINDOW = 50
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class TrainingStatsCallback(BaseCallback):
    """Records score per episode; live Score vs Episode plot; saves JSON/CSV/PNG at end."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        run_name: str | None = None,
        show_live: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.run_name = run_name or "training_run"
        self.show_live = show_live
        self.episode_scores: list[int] = []
        self.timesteps_at_episode: list[int] = []
        self._fig, self._ax = None, None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if not isinstance(infos, list):
            infos = [infos] if infos else []
        if not isinstance(dones, list):
            dones = [dones] if dones else []
        for i, done in enumerate(dones):
            if not done or i >= len(infos) or "episode" not in infos[i]:
                continue
            info, ep = infos[i], infos[i]["episode"]
            r = float(ep.get("r", 0))
            self.timesteps_at_episode.append(self.num_timesteps)
            score = info.get("score")
            if score is not None:
                try:
                    self.episode_scores.append(int(score))
                except (TypeError, ValueError):
                    self.episode_scores.append(max(0, int(round((r + 100) / 10.0))))
            else:
                self.episode_scores.append(0 if r <= -100 else max(0, int(round((r + 100) / 10.0))))
        return True

    def _on_rollout_end(self) -> None:
        if self.show_live:
            self._update_plot()

    def _update_plot(self) -> None:
        if self._fig is None:
            if self.show_live:
                plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 5))
            if self.show_live:
                plt.show(block=False)
        ax = self._ax
        ax.clear()
        if self.episode_scores:
            n, s = len(self.episode_scores), self.episode_scores
            ax.plot(range(1, n + 1), s, color="steelblue", alpha=0.4, label="Score")
            if n >= ROLLING_WINDOW:
                ma = np.convolve(s, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
                ax.plot(range(ROLLING_WINDOW, n + 1), ma, color="darkblue", lw=2, label=f"Moving avg (w={ROLLING_WINDOW})")
        ax.set(xlabel="Episode", ylabel="Score", title="Score vs Episode")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._fig.canvas.draw_idle()
        if self.show_live:
            plt.pause(0.001)

    def _on_training_end(self) -> None:
        self._update_plot()
        self._save_stats()

    def _save_stats(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / self.run_name
        scores = self.episode_scores
        times = self.timesteps_at_episode

        # JSON
        a = np.array(scores, dtype=float) if scores else np.array([])
        summary = {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()), "count": len(a)} if len(a) else {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"run_name": self.run_name, "total_timesteps": int(self.num_timesteps), "episode_scores": [int(x) for x in scores], "timesteps_at_episode": [int(t) for t in times], "score_summary": summary}, f, indent=2)
        if self.verbose:
            print(f"Stats saved to {base.with_suffix('.json')}")

        # CSV
        with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["episode", "score", "timestep"])
            for i in range(len(scores)):
                w.writerow([i + 1, scores[i], times[i] if i < len(times) else ""])
        if self.verbose:
            print(f"CSV saved to {base.with_suffix('.csv')}")

        # PNG
        if self._fig is not None:
            self._fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"Figure saved to {base.with_suffix('.png')}")
            plt.close(self._fig)
            self._fig, self._ax = None, None
