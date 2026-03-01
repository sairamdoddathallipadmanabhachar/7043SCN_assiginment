import os
import csv
from stable_baselines3 import PPO
from agents.chefhat_env import ChefHatEnv


def evaluate(model_path, opponent_type="random", n_games=200):

    env = ChefHatEnv(opponent_type=opponent_type)
    model = PPO.load(model_path)

    wins = 0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)

        if reward == 1:
            wins += 1

    win_rate = wins / n_games

    print(f"Model: {model_path}")
    print(f"Opponent: {opponent_type}")
    print(f"Win Rate: {win_rate}")

    # Save results
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.isfile("results/results.csv")

    with open("results/results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["model", "opponent", "win_rate"])

        writer.writerow([model_path, opponent_type, win_rate])

    return win_rate


if __name__ == "__main__":
    evaluate("results/models/ppo_opponent_model", opponent_type="heuristic")