from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO

from env.flappy_bird_env import FlappyBirdEnv
from training_stats import TrainingStatsCallback


def train_flappy_bird(
    timesteps: int = 100_000,
    render: bool = False,
    stats_output_dir: Path | str | None = None,
    run_name: str | None = None,
    stats_show_live: bool = True,
):
    """Train Flappy Bird agent with PPO. Optionally shows live training stats; always saves them at the end."""
    print("Training Flappy Bird with PPO")
    print(f"Total timesteps: {timesteps}")
    print("-" * 40)

    # Create environment
    render_mode = "human" if render else None
    env = FlappyBirdEnv(render_mode=render_mode)

    # Stats callback: optional live plot + save at end
    name = run_name or f"flappy_training_{timesteps}"
    stats_callback = TrainingStatsCallback(
        run_name=name,
        output_dir=stats_output_dir,
        show_live=stats_show_live,
        verbose=1,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    # Train the model (with optional live stats and final save)
    if stats_show_live:
        print("Starting training... (training statistics will update live and save at the end)")
    else:
        print("Starting training... (training statistics will be saved at the end)")
    model.learn(total_timesteps=timesteps, callback=stats_callback)

    # Save the model
    model.save("flappy_model")
    print("Model saved as 'flappy_model'")

    env.close()
    return model


def play_trained_model(model_path: str = "flappy_model", episodes: int = 5):
    """Watch the trained PPO model play Flappy Bird."""
    print(f"Loading model: {model_path}")

    # Create environment with rendering
    env = FlappyBirdEnv(render_mode="human")

    # Load model
    model = PPO.load(model_path, env=env)

    print(f"Watching trained agent play {episodes} episodes...")
    print("Close the window to stop early")

    scores = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False

        print(f"Episode {episode + 1}:")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = info.get("score", 0)
        scores.append(score)
        print(f"Score: {score}")

    env.close()

    print("\nResults:")
    print(f"Average Score: {sum(scores) / len(scores):.2f}")
    print(f"Best Score: {max(scores)}")

    return scores


