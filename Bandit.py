"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class Bandit(ABC):
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.n_arms = len(probabilities)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_reward = 0
        self.total_regret = 0
        self.optimal_reward = max(probabilities)
        self.regret = []

    def pull(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def update(self, arm, reward):
        raise NotImplementedError("This method should be implemented by subclasses")

    def experiment(self, n_trials):
        raise NotImplementedError("This method should be implemented by subclasses")

    def report(self, results, algorithm):
        cumulative_reward = sum(results['rewards'])
        avg_regret = np.mean(results['regrets'])
        logger.info(f"{algorithm} - Total Reward: {cumulative_reward}")
        logger.info(f"{algorithm} - Average Regret: {avg_regret}")
        return {"Total Reward": cumulative_reward, "Average Regret": avg_regret}


#____________________________________________
class Visualization():
    """
        A class for visualizing the performance of bandit algorithms.
        """

    @staticmethod
    def plot1(title, regrets, rewards):
        """
               Visualize the cumulative rewards and regrets with linear and log scales.

               Args:
                   title (str): The title of the plot.
                   regrets (list): List of cumulative regrets.
                   rewards (list): List of cumulative rewards.
               """

        plt.figure(figsize=(12, 6))

        # Linear scale
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(rewards), label="Cumulative Rewards")
        plt.plot(np.cumsum(regrets), label="Cumulative Regret")
        plt.title(f"{title} (Linear Scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Values")
        plt.legend()
        plt.grid()

        # Log scale
        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(rewards), label="Cumulative Rewards")
        plt.plot(np.cumsum(regrets), label="Cumulative Regret")
        plt.yscale("log")
        plt.title(f"{title} (Log Scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Values (Log Scale)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot2(results):
        """
            Compare cumulative regrets of multiple algorithms.

            Args:
               results (dict): A dictionary where keys are algorithm names and values are results containing 'regrets'.
        """

        plt.figure(figsize=(10, 6))
        for algo, data in results.items():
            plt.plot(np.cumsum(data["regrets"]), label=f"{algo} Regret")
        plt.title("Comparison of Algorithms")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid()
        plt.show()


#--------------------------------------#

# Epsilon-Greedy
class EpsilonGreedy(Bandit):
    """
        Implements the Epsilon-Greedy algorithm with decaying epsilon.
    """

    def __init__(self, probabilities, epsilon=1.0):
        """
            Initialize Epsilon-Greedy with probabilities and initial epsilon.

            Args:
                probabilities (list): Probabilities for each arm.
                epsilon (float): Initial exploration rate.
        """

        super().__init__(probabilities)
        self.epsilon = epsilon

    def pull(self):
        """
            Select an arm to pull using epsilon-greedy strategy.

            Returns:
                int: Index of the selected arm.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        return np.argmax(self.values)

    def update(self, arm, reward):
        """
            Update the bandit's state after pulling an arm.

            Args:
                arm (int): The index of the arm pulled.
                reward (float): The reward obtained.
        """
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.total_reward += reward
        self.total_regret += self.optimal_reward - self.probabilities[arm]
        self.regret.append(self.total_regret)

    def experiment(self, n_trials):

        """
           Run the Epsilon-Greedy algorithm for a specified number of trials.

           Args:
               n_trials (int): Number of trials to run.

           Returns:
                dict: Results with keys 'rewards' and 'regrets'.
        """

        rewards = []
        regrets = []
        for t in range(1, n_trials + 1):
            self.epsilon = 1 / t  # Decaying epsilon
            arm = self.pull()
            reward = 1 if np.random.rand() < self.probabilities[arm] else 0
            self.update(arm, reward)
            rewards.append(reward)
            regrets.append(self.total_regret)
        return {"rewards": rewards, "regrets": regrets}


# Thompson Sampling
class ThompsonSampling(Bandit):

    """
        Implements the Thompson Sampling algorithm for bandit problems.

        Attributes:
            alpha (np.array): Success counts for each arm.
            beta (np.array): Failure counts for each arm.
    """

    def __init__(self, probabilities):
        """
            Initialize Thompson Sampling with probabilities.

           Args:
               probabilities (list): Probabilities for each arm.
        """
        super().__init__(probabilities)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def pull(self):
        """
            Select an arm to pull using Thompson Sampling strategy.

            Returns:
                int: Index of the selected arm.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        """
            Update the bandit's state after pulling an arm.

            Args:
                arm (int): The index of the arm pulled.
                reward (float): The reward obtained.
        """

        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.total_reward += reward
        self.total_regret += self.optimal_reward - self.probabilities[arm]
        self.regret.append(self.total_regret)

    def experiment(self, n_trials):
        """
            Run the Thompson Sampling algorithm for a specified number of trials.

            Args:
                n_trials (int): Number of trials to run.

            Returns:
                dict: Results with keys 'rewards' and 'regrets'.
        """
        rewards = []
        regrets = []
        for _ in range(n_trials):
            arm = self.pull()
            reward = 1 if np.random.rand() < self.probabilities[arm] else 0
            self.update(arm, reward)
            rewards.append(reward)
            regrets.append(self.total_regret)
        return {"rewards": rewards, "regrets": regrets}


#--------------------------------------#

def comparison(results):
    """
        Compare the performance metrics of multiple bandit algorithms.

        This function calculates and logs the cumulative rewards, average regrets,
        maximum cumulative rewards, and minimum cumulative regrets for each algorithm.

        Args:
            results (dict): A dictionary where keys are algorithm names (str) and
                            values are dictionaries containing:
                                - "rewards" (list): Rewards collected at each trial.
                                - "regrets" (list): Regrets accumulated at each trial.
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




if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    logger.info("Starting Bandit Experiments")

    bandit_reward = [0.1, 0.2, 0.3, 0.4]
    n_trials = 20000

    # Epsilon-Greedy Experiment
    eg = EpsilonGreedy(bandit_reward)
    eg_results = eg.experiment(n_trials)
    eg_report = eg.report(eg_results, "Epsilon-Greedy")

    # Thompson Sampling Experiment
    ts = ThompsonSampling(bandit_reward)
    ts_results = ts.experiment(n_trials)
    ts_report = ts.report(ts_results, "Thompson Sampling")

    # Visualize Learning Curves
    Visualization.plot1("Epsilon-Greedy Learning Curve", eg_results["regrets"], eg_results["rewards"])
    Visualization.plot1("Thompson Sampling Learning Curve", ts_results["regrets"], ts_results["rewards"])

    # Compare Algorithms
    results = {
        "Epsilon-Greedy": eg_results,
        "Thompson Sampling": ts_results
    }
    Visualization.plot2(results)
    comparison(results)

    # Save results to CSV
    df = pd.DataFrame({
        "Epsilon-Greedy Rewards": eg_results["rewards"],
        "Epsilon-Greedy Regrets": eg_results["regrets"],
        "Thompson Sampling Rewards": ts_results["rewards"],
        "Thompson Sampling Regrets": ts_results["regrets"]
    })
    df.to_csv("bandit_results.csv", index=False)
    logger.info("Results saved to 'bandit_results.csv'")

