from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.
    Handles probabilities, reward tracking, and cumulative regret computation.
    """

    def __init__(self, probabilities: list[float]):
        """
        Initialize the bandit with arm probabilities.

        Args:
            probabilities (list[float]): Probabilities for each arm.
        """
        self.probabilities = probabilities
        self.n_arms = len(probabilities)
        self.total_reward = 0
        self.total_regret = 0
        self.optimal_reward = max(probabilities)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.regrets = []
        self.rewards = []

    @abstractmethod
    def pull(self) -> int:
        """Select an arm to pull. Must be implemented by subclasses."""
        pass

    def update(self, arm: int, reward: int):
        """
        Update the bandit's internal state based on the reward.

        Args:
            arm (int): The index of the pulled arm.
            reward (int): The reward obtained (1 or 0).
        """
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.total_reward += reward
        self.total_regret += self.optimal_reward - self.probabilities[arm]
        self.regrets.append(self.total_regret)
        self.rewards.append(reward)

    def experiment(self, n_trials: int):
        """
        Run the experiment for a given number of trials.

        Args:
            n_trials (int): Number of trials to run.
        """
        for _ in range(n_trials):
            arm = self.pull()
            reward = 1 if np.random.rand() < self.probabilities[arm] else 0
            self.update(arm, reward)

class EpsilonGreedy(Bandit):
    """
    Implements the Epsilon-Greedy algorithm with decaying epsilon.
    """

    def __init__(self, probabilities: list[float], epsilon: float = 1.0):
        """
        Initialize Epsilon-Greedy.

        Args:
            probabilities (list[float]): Probabilities for each arm.
            epsilon (float): Initial exploration rate.
        """
        super().__init__(probabilities)
        self.epsilon = epsilon

    def pull(self) -> int:
        """
        Select an arm to pull using epsilon-greedy strategy.

        Returns:
            int: Index of the selected arm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        return np.argmax(self.values)

    def experiment(self, n_trials: int):
        """
        Override experiment to implement decaying epsilon.
        """
        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t
            super().experiment(1)
class ThompsonSampling(Bandit):
    """
    Implements the Thompson Sampling algorithm for bandit problems.
    """

    def __init__(self, probabilities: list[float]):
        """
        Initialize Thompson Sampling.

        Args:
            probabilities (list[float]): Probabilities for each arm.
        """
        super().__init__(probabilities)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def pull(self) -> int:
        """
        Select an arm to pull using Thompson Sampling.

        Returns:
            int: Index of the selected arm.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm: int, reward: int):
        """
        Update the bandit's state with Bayesian priors.

        Args:
            arm (int): The index of the pulled arm.
            reward (int): The reward obtained (1 or 0).
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        super().update(arm, reward)
class Visualization:
    """
    Visualization tools for bandit performance metrics.
    """

    @staticmethod
    def compare(results: dict):
        """
        Compare cumulative rewards and regrets for multiple algorithms.

        Args:
            results (dict): A dictionary of results for each algorithm.
        """
        plt.figure(figsize=(12, 6))
        for algo, data in results.items():
            plt.plot(np.cumsum(data["rewards"]), label=f"{algo} Rewards")
            plt.plot(np.cumsum(data["regrets"]), label=f"{algo} Regrets")
        plt.title("Algorithm Comparison")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Values")
        plt.legend()
        plt.grid()
        plt.show()

probabilities = [0.1, 0.2, 0.3, 0.4]

def run_experiment(algorithm, name, n_trials):
    """
    Run a bandit experiment for a specific algorithm.

    Args:
        algorithm (class): The bandit algorithm class (e.g., EpsilonGreedy, ThompsonSampling).
        name (str): Name of the algorithm.
        n_trials (int): Number of trials to run.

    Returns:
        dict: A dictionary containing rewards and regrets.
    """
    algo = algorithm(probabilities)
    algo.experiment(n_trials)
    return {"rewards": algo.rewards, "regrets": algo.regrets}

def comparison(results):
    """
    Compare cumulative rewards and regrets for multiple algorithms.

    Args:
        results (dict): A dictionary of results for each algorithm, where keys are algorithm names and
                        values are dictionaries with "rewards" and "regrets".

    Returns:
        pd.DataFrame: A DataFrame summarizing key metrics for each algorithm.
    """
    comparison_metrics = []

    for algo, data in results.items():
        cumulative_reward = sum(data["rewards"])
        avg_regret = np.mean(data["regrets"])
        max_reward = max(np.cumsum(data["rewards"]))
        min_regret = min(np.cumsum(data["regrets"]))

        logger.info(f"{algo} - Total Reward: {cumulative_reward}")
        logger.info(f"{algo} - Average Regret: {avg_regret}")
        logger.info(f"{algo} - Maximum Cumulative Reward: {max_reward}")
        logger.info(f"{algo} - Minimum Cumulative Regret: {min_regret}")

        comparison_metrics.append({
            "Algorithm": algo,
            "Cumulative Reward": cumulative_reward,
            "Average Regret": avg_regret,
            "Maximum Cumulative Reward": max_reward,
            "Minimum Cumulative Regret": min_regret
        })


    comparison_df = pd.DataFrame(comparison_metrics)
    logger.info("\nComparison Metrics:\n" + comparison_df.to_string(index=False))
    return comparison_df

# Run experiments for Epsilon-Greedy and Thompson Sampling
results = {}
results["Epsilon-Greedy"] = run_experiment(EpsilonGreedy, "Epsilon-Greedy", 20000)
results["Thompson Sampling"] = run_experiment(ThompsonSampling, "Thompson Sampling", 20000)

# Perform comparison and save metrics
comparison_df = comparison(results)
comparison_df.to_csv("comparison_metrics.csv", index=False)
logger.info("Comparison metrics saved to 'comparison_metrics.csv'")

# Visualize the results
Visualization.compare(results)
