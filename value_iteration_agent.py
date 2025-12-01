"""
Agent utilisant Value Iteration pour résoudre le GridWorld
"""

import numpy as np
from typing import Tuple, Dict
from gridworld_env import GridWorldEnv


class RandomAgent:
    """
    Agent qui choisit des actions aléatoires
    """

    def __init__(self, env: GridWorldEnv):
        self.env = env

    def select_action(self, state: Tuple[int, int]) -> int:
        """
        Sélectionne une action aléatoire

        Args:
            state: État actuel

        Returns:
            Action aléatoire
        """
        return np.random.randint(0, self.env.action_space_n)


class ValueIterationAgent:
    """
    Agent utilisant l'algorithme Value Iteration
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, theta: float = 1e-6):
        """
        Initialise l'agent Value Iteration

        Args:
            env: Environnement GridWorld
            gamma: Facteur de discount
            theta: Seuil de convergence
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Initialiser la table de valeurs
        self.V = np.zeros((env.rows, env.cols))

        # Politique optimale
        self.policy = {}

        # Noms des actions
        self.action_names = {
            GridWorldEnv.UP: "↑",
            GridWorldEnv.DOWN: "↓",
            GridWorldEnv.LEFT: "←",
            GridWorldEnv.RIGHT: "→"
        }

    def value_iteration(self, max_iterations: int = 1000) -> int:
        """
        Exécute l'algorithme Value Iteration

        Args:
            max_iterations: Nombre maximum d'itérations

        Returns:
            Nombre d'itérations effectuées
        """
        iterations = 0

        for i in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)

            # Pour chaque état
            for row in range(self.env.rows):
                for col in range(self.env.cols):
                    state = (row, col)

                    # Ignorer les obstacles et le goal
                    if state in self.env.obstacles:
                        continue

                    if self.env.is_terminal_state(state):
                        new_V[row, col] = 0
                        continue

                    # Calculer la valeur maximale parmi toutes les actions
                    action_values = []
                    for action in range(self.env.action_space_n):
                        next_state = self.env.get_next_state(state, action)
                        reward = self.env.get_reward(state, action, next_state)
                        next_row, next_col = next_state
                        value = reward + self.gamma * self.V[next_row, next_col]
                        action_values.append(value)

                    new_V[row, col] = max(action_values)

                    # Calculer le changement maximum
                    delta = max(delta, abs(self.V[row, col] - new_V[row, col]))

            self.V = new_V
            iterations += 1

            # Vérifier la convergence
            if delta < self.theta:
                break

        # Extraire la politique optimale
        self._extract_policy()

        return iterations

    def _extract_policy(self):
        """
        Extrait la politique optimale à partir de la table de valeurs
        """
        self.policy = {}

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                state = (row, col)

                # Ignorer les obstacles et le goal
                if state in self.env.obstacles or self.env.is_terminal_state(state):
                    continue

                # Trouver la meilleure action
                best_action = None
                best_value = float('-inf')

                for action in range(self.env.action_space_n):
                    next_state = self.env.get_next_state(state, action)
                    reward = self.env.get_reward(state, action, next_state)
                    next_row, next_col = next_state
                    value = reward + self.gamma * self.V[next_row, next_col]

                    if value > best_value:
                        best_value = value
                        best_action = action

                self.policy[state] = best_action

    def select_action(self, state: Tuple[int, int]) -> int:
        """
        Sélectionne une action selon la politique optimale

        Args:
            state: État actuel

        Returns:
            Action optimale
        """
        if state in self.policy:
            return self.policy[state]
        else:
            # Si pas de politique pour cet état, action aléatoire
            return np.random.randint(0, self.env.action_space_n)

    def get_value_grid(self) -> np.ndarray:
        """
        Retourne la grille de valeurs

        Returns:
            Grille de valeurs
        """
        return self.V.copy()

    def get_policy_grid(self) -> Dict[Tuple[int, int], str]:
        """
        Retourne la politique sous forme de grille avec symboles

        Returns:
            Dictionnaire état -> symbole d'action
        """
        policy_grid = {}
        for state, action in self.policy.items():
            policy_grid[state] = self.action_names[action]
        return policy_grid
