from pyvirtualdisplay import Display
import numpy as np
import gym
import random
import imageio
import logging
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="gym.wrappers.monitoring.video_recorder",
)
warnings.filterwarnings("ignore")


class TrainFrozenAgent:
    def __init__(self) -> None:
        pass

    def environment(self, env_name: str) -> any:
        """
        Create the FrozenLake-v1 environment using 4x4 map and non-slippery version

        Parameters:
        env_name (type): Name of Environement(include on Gym library).
        ...

        Returns:
            Any: An RL Environment.
        """
        logging.info("creating environment")
        env = gym.make(env_name)
        self.env_name = env_name

        return env

    def epsilon_greedy_policy(
        self, Qtable: np.ndarray, state: int, epsilon: np.float64
    ) -> np.int64:
        """
        Epsilon Greedy Policy implementation

        Parameters:
        Qtable (numpy.ndarray): Qtable by default a zero table.
        state (int): State of Environement.
        epsilon (np.float64): Epsilon value.
        ...

        Returns:
            np.int64: An action to be taken by agent.
        """

        env = self.environment(self.env_name)
        # Randomly generate a number between 0 and 1
        random_int = random.uniform(0, 1)
        # if random_int > greater than epsilon --> exploitation
        if random_int > epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = np.argmax(Qtable[state])
        # else --> exploration
        else:
            action = env.action_space.sample()

        return action

    def q_table(self) -> any:
        """
        Q learning algorithm from scratch
        it's following a Q-learning paper

        Parameters:
            None
        ...

        Returns:
            np.ndarray: Null matrix of QTable.
        """

        # environment
        env = self.environment(self.env_name)

        # state space and action space (of environment)
        state_space = env.observation_space.n
        action_space = env.action_space.n

        # create a Q-Table
        Qtable = np.zeros((state_space, action_space))

        return Qtable

    def train(
        self,
        n_training_episodes: int,
        learning_rate: float,
        max_steps: int,
        gamma: float,
        max_epsilon: float,
        min_epsilon: float,
        decay_rate: float,
    ) -> np.ndarray:
        """
        Initialize the Training Algorithm (Q-learning)

        Parameters:
            n_training_episodes (int): Number of the training episodes to run.
            learning_rate (float): Learning rate value.
            max_steps (int): Max steps to trainning.
            gamma (float): Gamma value ((+) exploration | (-) exploitation).
            max_epsilon (float): The maximum value of epsilon.
            min_epsilon (float): The minimum value of epsilon.
            decay_rate (float): Decay rate value (forget past rewards).
        ...

        Returns:
            np.ndarray: Trained QTable.
        """

        # set up environment
        env = self.environment(self.env_name)
        # get Qtable
        Qtable = self.q_table()

        for episode in tqdm(range(n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
                -decay_rate * episode
            )
            # Reset the environment
            state = env.reset()
            step = 0
            done = False

            # repeat
            for step in range(max_steps):
                # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(Qtable, state, epsilon)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                Qtable[state][action] = Qtable[state][action] + learning_rate * (
                    reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
                )

                # If done, finish the episode
                if done:
                    break

                # Our state is the new state
                state = new_state
                self.env = env
        self.Qtable = Qtable
        return Qtable

    def virtual_display(self):
        """
        A virtual display to show agent on environment

        Args:
           None

        Return:
           None

        """
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    def record_video(self, out_directory):
        """
        Records a video of an agent interacting with an environment based on a Q-table.

        Parameters:
        - out_directory (str): The path to the directory where the output video will be saved.

        Returns:
        None

        Example:
        >> record_video(env, Qtable, './video.gif', fps=5)
        """

        self.virtual_display()
        images = []
        done = False
        state = self.env.reset(seed=random.randint(0, 500))
        img = self.env.render(mode="rgb_array")
        images.append(img)
        while not done:
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(self.Qtable[state][:])
            state, reward, done, info = self.env.step(
                action
            )  # We directly put next_state = state for recording logic
            img = self.env.render(mode="rgb_array")
            images.append(img)
        imageio.mimsave(
            out_directory, [np.array(img) for i, img in enumerate(images)], fps=1
        )

    def evaluation(self, max_steps, n_eval_episodes, Q, out_directory, num_episodes):
        """Evaluate the agent

        Args:
           max_steps: The max steps to run
           n_eval_episodes: Number of episode to evaluate the agent
           Q: The Q-table
           out_directory (str): The path to the directory where the output video will be saved.


        Return:
           float: std and mean of reward
        """
        self.record_video(out_directory)
        episode_rewards = []
        seed = []
        for episode in tqdm(range(n_eval_episodes)):
            if seed:
                state = self.env.reset(seed=seed[episode])
            else:
                state = self.env.reset()
                step = 0
                done = False
                total_rewards_ep = 0

            for step in range(max_steps):
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(Q[state][:])
                new_state, reward, done, info = self.env.step(action)
                total_rewards_ep += reward

                if done:
                    break
            state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        description="Traing a RL Agent on FrozenLake Environment"
    )
    parser.add_argument("--env_name", type=str, help="name of gym environment")
    parser.add_argument("--map_name", type=str, help="a map size of gym environment")
    parser.add_argument(
        "--n_training_episodes", type=int, help="number of episodes to run"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate of RL algorithm"
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, help="number of episodes to evaluation"
    )
    parser.add_argument("--max_steps", type=int, help="number of steps")
    parser.add_argument(
        "--gamma", type=float, help="gamma value (exploration x exploitation)"
    )
    parser.add_argument("--max_epsilon", type=float, help="maximum value of epsilon")
    parser.add_argument("--min_epsilon", type=float, help="minimum value of epsilon")
    parser.add_argument("--decay_rate", type=float, help="decay rate")
    parser.add_argument(
        "--out_directory", type=str, help="Directory to save agent video"
    )
    parser.add_argument("--num_episodes", type=int, help="Number of videos to be saved")

    args = parser.parse_args()

    # RL configuration
    model_rl = TrainFrozenAgent()
    model_rl.environment(env_name=args.env_name)
    Qtable = model_rl.train(
        n_training_episodes=args.n_training_episodes,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gamma=args.gamma,
        max_epsilon=args.max_epsilon,
        min_epsilon=args.min_epsilon,
        decay_rate=args.decay_rate,
    )
    mean_reward, std_reward = model_rl.evaluation(
        max_steps=args.max_steps,
        n_eval_episodes=args.n_eval_episodes,
        Q=Qtable,
        out_directory=args.out_directory,
        num_episodes=args.num_episodes,
    )
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
