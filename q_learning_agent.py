"""
Agent Q-Learning pour apprentissage par renforcement RÉEL
L'agent apprend par expérience (essais/erreurs)
"""

import numpy as np
from typing import Tuple, Dict
from gridworld_env import GridWorldEnv


class QLearningAgent:
    """
    Agent Q-Learning qui apprend par expérience
    """

    def __init__(self, env: GridWorldEnv,
                 learning_rate: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialise l'agent Q-Learning

        Args:
            env: Environnement GridWorld
            learning_rate: Taux d'apprentissage (alpha)
            gamma: Facteur de discount
            epsilon: Taux d'exploration initial
            epsilon_decay: Décroissance d'epsilon
            epsilon_min: Epsilon minimum
        """
        self.env = env
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialiser la Q-table à zéro
        # Q[state][action] = valeur Q
        self.Q = {}
        for row in range(env.rows):
            for col in range(env.cols):
                state = (row, col)
                if state not in env.obstacles:
                    self.Q[state] = [0.0] * env.action_space_n

        # Statistiques d'apprentissage
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        self.training_complete = False

        # Noms des actions
        self.action_names = {
            GridWorldEnv.UP: "↑",
            GridWorldEnv.DOWN: "↓",
            GridWorldEnv.LEFT: "←",
            GridWorldEnv.RIGHT: "→"
        }

    def select_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Sélectionne une action avec epsilon-greedy

        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon greedy

        Returns:
            Action sélectionnée
        """
        if state not in self.Q:
            return np.random.randint(0, self.env.action_space_n)

        # Exploration vs Exploitation
        if training and np.random.random() < self.epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.env.action_space_n)
        else:
            # Exploitation : meilleure action selon Q
            return int(np.argmax(self.Q[state]))

    def update_q_value(self, state: Tuple[int, int], action: int,
                      reward: float, next_state: Tuple[int, int], done: bool):
        """
        Met à jour la Q-value avec la règle de Q-Learning

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Si l'épisode est terminé
        """
        if state not in self.Q:
            return

        # Valeur actuelle
        current_q = self.Q[state][action]

        # Meilleure valeur Q pour l'état suivant
        if done or next_state not in self.Q:
            max_next_q = 0.0
        else:
            max_next_q = max(self.Q[next_state])

        # Nouvelle valeur Q (règle de Q-Learning)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Mettre à jour
        self.Q[state][action] = new_q

    def train_episode(self, max_steps: int = 100, verbose: bool = False) -> Tuple[float, int]:
        """
        Entraîne l'agent sur un épisode

        Args:
            max_steps: Nombre maximum de steps
            verbose: Afficher les détails

        Returns:
            (récompense totale, nombre de steps)
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0

        while not self.env.done and steps < max_steps:
            # Sélectionner une action
            action = self.select_action(state, training=True)

            # Exécuter l'action
            next_state, reward, done, info = self.env.step(action)

            # Mettre à jour la Q-value
            self.update_q_value(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

            if verbose and steps % 10 == 0:
                print(f"  Step {steps}: pos={state}, reward={total_reward:.3f}")

        # Décroissance d'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_reward, steps

    def train(self, num_episodes: int = 1000, max_steps: int = 100,
              verbose_every: int = 100, visualize_callback=None,
              visualize_every: int = 50) -> Dict:
        """
        Entraîne l'agent sur plusieurs épisodes

        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum de steps par épisode
            verbose_every: Afficher les stats tous les N épisodes
            visualize_callback: Fonction de visualisation (env, agent, episode, total)
            visualize_every: Visualiser tous les N épisodes

        Returns:
            Statistiques d'entraînement
        """
        print(f"Début de l'entraînement sur {num_episodes} épisodes...")
        if visualize_callback:
            print(f"Visualisation activée tous les {visualize_every} épisodes")

        for episode in range(num_episodes):
            # Visualiser certains épisodes
            if visualize_callback and (episode + 1) % visualize_every == 0:
                print(f"\n>>> Visualisation de l'épisode {episode + 1}...")
                total_reward, steps = visualize_callback(self.env, self, episode + 1,
                                                         num_episodes, max_steps)
            else:
                total_reward, steps = self.train_episode(max_steps)

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)

            # Afficher les stats
            if (episode + 1) % verbose_every == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-verbose_every:])
                avg_steps = np.mean(self.episode_steps[-verbose_every:])
                success_rate = sum([1 for r in self.episode_rewards[-verbose_every:] if r > 0]) / min(verbose_every, episode + 1)
                print(f"Épisode {episode + 1}/{num_episodes} | "
                      f"Récompense moy: {avg_reward:.3f} | "
                      f"Steps moy: {avg_steps:.1f} | "
                      f"Succès: {success_rate*100:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")

        self.training_complete = True
        print("Entraînement terminé!")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "epsilon_history": self.epsilon_history
        }

    def get_value_grid(self) -> np.ndarray:
        """
        Retourne la grille de valeurs (max Q-value pour chaque état)

        Returns:
            Grille de valeurs
        """
        V = np.zeros((self.env.rows, self.env.cols))

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                state = (row, col)
                if state in self.Q:
                    V[row, col] = max(self.Q[state])

        return V

    def get_policy_grid(self) -> Dict[Tuple[int, int], str]:
        """
        Retourne la politique apprise sous forme de grille avec symboles

        Returns:
            Dictionnaire état -> symbole d'action
        """
        policy_grid = {}

        for state in self.Q:
            if state not in self.env.obstacles and state != self.env.goal_pos:
                best_action = int(np.argmax(self.Q[state]))
                policy_grid[state] = self.action_names[best_action]

        return policy_grid

    def reset_for_new_goal(self):
        """
        Réinitialise partiellement l'agent pour un nouveau goal
        Garde l'expérience acquise mais réinitialise epsilon pour explorer
        """
        # Ne pas réinitialiser complètement la Q-table
        # L'agent garde sa connaissance de la navigation
        # Mais on augmente légèrement epsilon pour explorer le nouveau goal
        self.epsilon = min(0.3, self.epsilon * 2)
        print(f"    Agent adapté pour nouveau goal (epsilon ajusté à {self.epsilon:.3f})")
