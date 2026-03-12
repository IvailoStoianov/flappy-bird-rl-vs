from __future__ import annotations

from stable_baselines3 import PPO

from env.flappy_bird_env import FlappyBirdEnv


def train_flappy_bird(timesteps: int = 100_000, render: bool = False):
    """Train Flappy Bird agent with PPO."""
    print("Training Flappy Bird with PPO")
    print(f"Total timesteps: {timesteps}")
    print("-" * 40)

    # Create environment
    render_mode = "human" if render else None
    env = FlappyBirdEnv(render_mode=render_mode)

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

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=timesteps)

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


def main():
    """Simple CLI for training and playing the Flappy Bird game."""
    import sys

    if len(sys.argv) == 1:
        # Default training
        train_flappy_bird()
    elif sys.argv[1] == "train":
        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
        render = "--render" in sys.argv
        train_flappy_bird(timesteps, render)
    elif sys.argv[1] == "play":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "flappy_model"
        episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        play_trained_model(model_path, episodes)
    else:
        print("Usage:")
        print("  python -m training.train_flappy_bird                 # Train with defaults")
        print("  python -m training.train_flappy_bird train 200000    # Train for 200k steps")
        print(
            "  python -m training.train_flappy_bird train 50000 --render  # Train with rendering"
        )
        print("  python -m training.train_flappy_bird play            # Watch trained model")
        print(
            "  python -m training.train_flappy_bird play flappy_model 3   # Watch model play 3 games"
        )


if __name__ == "__main__":
    main()


