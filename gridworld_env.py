"""
Environnement GridWorld pour Reinforcement Learning
Inspiré de Gymnasium (OpenAI Gym)
"""

import numpy as np
from typing import Tuple, Optional


class GridWorldEnv:
    """
    Environnement GridWorld simple pour le Reinforcement Learning

    L'agent doit naviguer dans une grille pour atteindre le goal en évitant les obstacles.
    """

    # Actions possibles
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, grid_size: Tuple[int, int] = (5, 5),
                 start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (4, 4),
                 obstacles: list = None):
        """
        Initialise l'environnement GridWorld

        Args:
            grid_size: Taille de la grille (rows, cols)
            start_pos: Position initiale (row, col)
            goal_pos: Position du goal (row, col)
            obstacles: Liste des positions d'obstacles [(row, col), ...]
        """
        self.rows, self.cols = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles if obstacles is not None else []

        # État actuel
        self.current_pos = None
        self.done = False

        # Paramètres de récompense
        self.goal_reward = 1.0
        self.obstacle_penalty = -1.0
        self.step_penalty = -0.01

        # Nombre d'actions possibles
        self.action_space_n = 4
        self.observation_space_n = self.rows * self.cols

    def reset(self) -> Tuple[int, int]:
        """
        Réinitialise l'environnement

        Returns:
            Position initiale
        """
        self.current_pos = self.start_pos
        self.done = False
        return self.current_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Exécute une action dans l'environnement

        Args:
            action: Action à exécuter (UP, DOWN, LEFT, RIGHT)

        Returns:
            next_state: Nouvel état
            reward: Récompense obtenue
            done: Si l'épisode est terminé
            info: Informations supplémentaires
        """
        if self.done:
            return self.current_pos, 0.0, True, {}

        # Calculer la nouvelle position
        row, col = self.current_pos

        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.DOWN:
            row = min(self.rows - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        elif action == self.RIGHT:
            col = min(self.cols - 1, col + 1)

        new_pos = (row, col)

        # Calculer la récompense
        reward = self.step_penalty

        # Vérifier si on atteint le goal
        if new_pos == self.goal_pos:
            reward = self.goal_reward
            self.done = True
        # Vérifier si on touche un obstacle
        elif new_pos in self.obstacles:
            reward = self.obstacle_penalty
            # On reste à la position actuelle si on touche un obstacle
            new_pos = self.current_pos

        self.current_pos = new_pos

        info = {"position": self.current_pos}

        return self.current_pos, reward, self.done, info

    def get_all_states(self) -> list:
        """
        Retourne tous les états possibles

        Returns:
            Liste de tous les états (positions)
        """
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) not in self.obstacles:
                    states.append((row, col))
        return states

    def is_terminal_state(self, state: Tuple[int, int]) -> bool:
        """
        Vérifie si un état est terminal

        Args:
            state: État à vérifier

        Returns:
            True si l'état est terminal, False sinon
        """
        return state == self.goal_pos

    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Retourne le prochain état sans modifier l'environnement
        Utile pour la planification (Value Iteration, Policy Iteration)

        Args:
            state: État actuel
            action: Action à exécuter

        Returns:
            Prochain état
        """
        row, col = state

        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.DOWN:
            row = min(self.rows - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        elif action == self.RIGHT:
            col = min(self.cols - 1, col + 1)

        new_pos = (row, col)

        # Si obstacle, on reste à la position actuelle
        if new_pos in self.obstacles:
            return state

        return new_pos

    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        """
        Retourne la récompense pour une transition

        Args:
            state: État actuel
            action: Action exécutée
            next_state: État suivant

        Returns:
            Récompense
        """
        if next_state == self.goal_pos:
            return self.goal_reward
        elif next_state in self.obstacles:
            return self.obstacle_penalty
        else:
            return self.step_penalty

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """
        Convertit un état (row, col) en index

        Args:
            state: État (row, col)

        Returns:
            Index de l'état
        """
        row, col = state
        return row * self.cols + col

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """
        Convertit un index en état (row, col)

        Args:
            index: Index de l'état

        Returns:
            État (row, col)
        """
        row = index // self.cols
        col = index % self.cols
        return (row, col)

    def set_goal(self, new_goal: Tuple[int, int]):
        """
        Change la position du goal

        Args:
            new_goal: Nouvelle position du goal (row, col)
        """
        if new_goal in self.obstacles or new_goal == self.start_pos:
            raise ValueError("Le goal ne peut pas être sur un obstacle ou la position de départ")
        self.goal_pos = new_goal

    def get_random_goal(self) -> Tuple[int, int]:
        """
        Génère une position aléatoire valide pour le goal

        Returns:
            Position aléatoire du goal (row, col)
        """
        valid_positions = []
        for row in range(self.rows):
            for col in range(self.cols):
                pos = (row, col)
                if pos not in self.obstacles and pos != self.start_pos:
                    valid_positions.append(pos)

        if not valid_positions:
            raise ValueError("Aucune position valide pour le goal")

        return valid_positions[np.random.randint(len(valid_positions))]
