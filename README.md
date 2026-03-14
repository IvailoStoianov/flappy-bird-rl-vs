# Flappy Bird RL

A small reinforcement learning project: train an agent to play Flappy Bird using **PPO** (Proximal Policy Optimization) from Stable-Baselines3. The game is built with Pygame and wrapped as a Gymnasium environment.

## What it does

- **Train** a PPO agent on the Flappy Bird game (no rendering by default; optional live stats plot).
- **Watch** a trained agent play (render the game and see the model control the bird).
- **Play** the game yourself with the space bar.

Training records **score (pipes passed) per episode** and saves a **Score vs Episode** plot (with moving average) plus JSON/CSV at the end (and optionally live during training).

## How it works

1. **Game** (`game/`) – Flappy Bird implemented in Pygame (bird, pipes, collision, score).
2. **Environment** (`env/`) – Gymnasium wrapper: 2 actions (flap / no flap), 6 normalized observations (bird position/velocity, pipe distances), rewards (+0.1 per step, +10 per pipe, −100 on death).
3. **Training** (`training/`) – PPO from Stable-Baselines3 trains on this env and saves a model (e.g. `flappy_model.zip`).
4. **Stats** (`training_stats/`) – Callback records score per episode, shows a live “Score vs Episode” plot (optional), and saves JSON/CSV/PNG when training ends.

## Setup (Conda)

Create and activate a conda environment, then install dependencies.

**Option A – use the environment file (recommended):**

```bash
conda env create -f environment.yml
conda activate flappy-bird-rl
```

**Option B – create the env and install by hand:**

```bash
conda create -n flappy-bird-rl python=3.11
conda activate flappy-bird-rl
pip install gymnasium stable-baselines3 pygame matplotlib numpy
```

Run everything from the **project root** (the folder that contains `main.py`).

**Pip only (no conda):** create a virtualenv, then `pip install -r requirements.txt`.

## CLI commands

All commands are run as:

```bash
python main.py <command> [options]
```

### Train the agent

```bash
python main.py train [--timesteps N] [--render] [--no-live-stats] [--stats-dir DIR] [--run-name NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--timesteps` | 100000 | Number of training steps. |
| `--render` | off | Show the game window while training (slower). |
| `--no-live-stats` | off | Don’t show the live Score vs Episode plot; only save stats at the end. |
| `--stats-dir` | `training_stats/output` | Folder for JSON/CSV/PNG stats. |
| `--run-name` | `flappy_training_<timesteps>` | Base name for stats files. |

**Examples:**

```bash
python main.py train
python main.py train --timesteps 500000 --no-live-stats
python main.py train --timesteps 100000 --render
```

After training, the model is saved as `flappy_model.zip` in the project root. Stats are written to the stats dir (e.g. `training_stats/output/flappy_training_100000.json`, `.csv`, `.png`).

### Watch the trained agent play

```bash
python main.py play-agent [--model PATH] [--episodes N]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | flappy_model | Path to the saved model (without `.zip`). |
| `--episodes` | 5 | Number of games to play. |

**Example:**

```bash
python main.py play-agent --episodes 10
```

### Play the game yourself

```bash
python main.py play-human
```

Use **space** to flap. Close the window to exit.

## Project layout

```
flappy-bird-rl/
├── main.py                 # CLI entry point
├── environment.yml         # Conda environment (dependencies)
├── game/                   # Flappy Bird (Pygame)
│   ├── flappy_bird.py
│   ├── bird.py
│   ├── pipe.py
│   └── assets/
├── env/
│   └── flappy_bird_env.py  # Gymnasium environment
├── training/
│   └── train_flappy_bird.py
├── training_stats/         # Score vs Episode stats and plot
│   ├── stats_callback.py
│   ├── output/             # Saved JSON/CSV/PNG (created when you train)
│   └── README.md
└── flappy_model.zip       # Saved model (created after training)
```

## Dependencies

- **Python** 3.10+
- **gymnasium** – RL environment API
- **stable-baselines3** – PPO implementation
- **pygame** – game rendering and input
- **numpy** – arrays and math
- **matplotlib** – training stats plot

These are listed in `environment.yml` for conda.
