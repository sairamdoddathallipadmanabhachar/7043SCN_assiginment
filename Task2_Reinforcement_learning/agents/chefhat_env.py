import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Chef's Hat source path (adjust if needed)
sys.path.append(r"C:\Users\saira\Downloads\ChefsHatGYM\src")

from core.game_env.game import Game
from core.logging.engine_logger import EngineLogger


class ChefHatEnv(gym.Env):
    """
    PPO-compatible single-agent wrapper for Chef's Hat.
    RL_Agent is controlled by PPO.
    Other players are random or heuristic.
    One full match = one episode.
    """

    def __init__(self, opponent_type="random"):
        super().__init__()

        self.player_names = ["RL_Agent", "Opponent1", "Opponent2", "Opponent3"]
        self.opponent_type = opponent_type

        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(200,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(200)

        self.total_games = 0
        self.total_wins = 0

        self.game = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        logger = EngineLogger(
            room_name="RLRoom",
            timestamp="training",
            player_names=self.player_names,
            config={},
            save_logs=False,
            output_folder="."
        )

        self.game = Game(
            player_names=self.player_names,
            max_matches=1,
            max_rounds=None,
            max_score=None,
            logger=logger,
            save_dataset=False,
        )

        self.game.start()
        self.game.deal_cards()
        self.game.create_new_match()
        self.game.start_match()

        obs = np.zeros(200, dtype=np.float32)
        return obs, {}

    def step(self, action):

        max_internal_steps = 500
        internal_counter = 0
        latest_observation = None

        while not self.game.finished and internal_counter < max_internal_steps:
            internal_counter += 1

            result = self.game.step()

            if result is None:
                break

            if result.get("request_action"):
                player = result["player"]
                observation = result["observation"]
                possible = observation["possible_actions"]

                if not possible:
                    continue

                latest_observation = observation

                # RL Agent action
                if player == "RL_Agent":
                    chosen_index = int(action) % len(possible)
                    chosen = possible[chosen_index]

                # Opponents
                else:
                    if self.opponent_type == "heuristic":
                        chosen = min(possible)
                    else:
                        chosen = random.choice(possible)

                result_after = self.game.step(chosen)

                if result_after.get("match_over"):
                    break

        if internal_counter >= max_internal_steps:
            self.game.finished = True

        reward = 0
        self.total_games += 1

        if self.game.finishing_order_last_game:
            if self.game.finishing_order_last_game[0] == "RL_Agent":
                reward = 1
                self.total_wins += 1

        # Use encoded state if available
        if latest_observation is not None:
            obs = self._encode_state(latest_observation)
        else:
            obs = np.zeros(200, dtype=np.float32)

        return obs, reward, True, False, {}

    def get_win_rate(self):
        if self.total_games == 0:
            return 0
        return self.total_wins / self.total_games

    def _encode_state(self, observation):
        """
        Encode hand and board into fixed vector.
        """

        hand = observation.get("hand", [])
        board = observation.get("board", [])

        hand_vec = np.zeros(13)
        board_vec = np.zeros(13)

        for card in hand:
            try:
                rank = int(card[0]) if isinstance(card, (list, tuple)) else int(card)
                if 0 <= rank < 13:
                    hand_vec[rank] += 1
            except:
                continue

        for card in board:
            try:
                rank = int(card[0]) if isinstance(card, (list, tuple)) else int(card)
                if 0 <= rank < 13:
                    board_vec[rank] += 1
            except:
                continue

        state = np.concatenate([hand_vec, board_vec])

        padded = np.zeros(200)
        padded[:len(state)] = state

        return padded.astype(np.float32)