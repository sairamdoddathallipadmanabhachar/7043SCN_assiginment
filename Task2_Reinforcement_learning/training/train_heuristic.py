import os
from agents.ppo_agent import create_agent
from agents.chefhat_env import ChefHatEnv


def main():
    os.makedirs("results/models", exist_ok=True)

    env = ChefHatEnv(opponent_type="heuristic")
    model = create_agent(env)

    model.learn(total_timesteps=256)
    model.save("results/models/ppo_heuristic")

    print("Final Win Rate:", env.get_win_rate())


if __name__ == "__main__":
    main()