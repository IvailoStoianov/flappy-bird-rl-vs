from __future__ import annotations

import argparse

from training.train_flappy_bird import train_flappy_bird, play_trained_model
from game.flappy_bird import main as play_human_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for Flappy Bird RL training and gameplay.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train RL agent
    train_parser = subparsers.add_parser("train", help="Train a PPO agent.")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of timesteps to train for (default: 100000).",
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while training.",
    )

    # Watch trained agent play
    play_agent_parser = subparsers.add_parser(
        "play-agent", help="Watch a trained PPO agent play."
    )
    play_agent_parser.add_argument(
        "--model",
        type=str,
        default="flappy_model",
        help="Path to the trained model (default: flappy_model).",
    )
    play_agent_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch (default: 5).",
    )

    # Play the game yourself
    subparsers.add_parser(
        "play-human",
        help="Play Flappy Bird yourself using the space bar.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_flappy_bird(timesteps=args.timesteps, render=bool(args.render))
    elif args.command == "play-agent":
        play_trained_model(model_path=args.model, episodes=args.episodes)
    elif args.command == "play-human":
        play_human_game()


if __name__ == "__main__":
    main()

