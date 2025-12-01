"""
Configuration pour le GridWorld et les agents
"""

# Paramètres de la grille
GRID_SIZE = (6, 6)
START_POS = (0, 0)
GOAL_POS = (5, 5)

# Obstacles (positions à éviter)
OBSTACLES = [
    (1, 1),
    (1, 2),
    (2, 3),
    (3, 3),
    (4, 1),
    (4, 2)
]

# Paramètres de l'agent Value Iteration
GAMMA = 0.95  # Facteur de discount
THETA = 1e-6  # Seuil de convergence
MAX_ITERATIONS = 1000  # Nombre maximum d'itérations

# Paramètres de l'agent Q-Learning (APPRENTISSAGE RÉEL)
USE_Q_LEARNING = True  # Si True, utilise Q-Learning au lieu de Value Iteration
LEARNING_RATE = 0.1  # Alpha - taux d'apprentissage
EPSILON_START = 1.0  # Exploration initiale (100%)
EPSILON_DECAY = 0.995  # Décroissance d'epsilon
EPSILON_MIN = 0.01  # Exploration minimale
NUM_TRAINING_EPISODES = 300  # Nombre d'épisodes d'entraînement (réduit pour démo)
VERBOSE_EVERY = 50  # Afficher stats tous les N épisodes

# Paramètres de test
NUM_TEST_EPISODES = 5  # Nombre d'épisodes de test
MAX_STEPS_PER_EPISODE = 100  # Nombre maximum de steps par épisode

# Paramètres de visualisation
CELL_SIZE = 1.0  # Taille d'une cellule
FONT_SIZE_VALUE = 10  # Taille de la police pour les valeurs
FONT_SIZE_POLICY = 14  # Taille de la police pour la politique
ANIMATION_DELAY = 0.3  # Délai entre chaque mouvement de l'agent (secondes)
NUM_ANIMATED_EPISODES = 3  # Nombre d'épisodes à animer

# Visualisation de l'entraînement
VISUALIZE_TRAINING = True  # Visualiser l'entraînement en temps réel
VISUALIZE_EVERY = 50  # Visualiser tous les N épisodes
TRAINING_ANIMATION_DELAY = 0.1  # Délai pendant l'entraînement (plus rapide)
SHOW_Q_VALUES_TRAINING = True  # Afficher les Q-values pendant l'entraînement

# Goal dynamique
DYNAMIC_GOAL = True  # Si True, le goal change à chaque épisode
SHOW_VALUE_CONVERGENCE = True  # Afficher la convergence lors du recalcul
