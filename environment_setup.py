"""
Configuration interactive de l'environnement GridWorld
"""

import numpy as np
from typing import Tuple, List
from gridworld_env import GridWorldEnv


def generate_random_obstacles(rows: int, cols: int, num_obstacles: int,
                              start_pos: Tuple[int, int],
                              goal_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Génère des obstacles aléatoires

    Args:
        rows: Nombre de lignes
        cols: Nombre de colonnes
        num_obstacles: Nombre d'obstacles à générer
        start_pos: Position de départ (à éviter)
        goal_pos: Position du goal (à éviter)

    Returns:
        Liste des positions d'obstacles
    """
    obstacles = []
    attempts = 0
    max_attempts = num_obstacles * 10

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        row = np.random.randint(0, rows)
        col = np.random.randint(0, cols)
        pos = (row, col)

        # Vérifier que ce n'est pas start, goal, ou déjà un obstacle
        if pos != start_pos and pos != goal_pos and pos not in obstacles:
            obstacles.append(pos)

        attempts += 1

    if len(obstacles) < num_obstacles:
        print(f"Attention: Seulement {len(obstacles)} obstacles générés sur {num_obstacles} demandés")

    return obstacles


def is_valid_configuration(rows: int, cols: int, num_obstacles: int) -> bool:
    """
    Vérifie si la configuration est valide

    Args:
        rows: Nombre de lignes
        cols: Nombre de colonnes
        num_obstacles: Nombre d'obstacles

    Returns:
        True si valide, False sinon
    """
    total_cells = rows * cols

    # Au minimum : 1 start + 1 goal + 1 chemin = 3 cellules libres
    min_free_cells = 3

    if num_obstacles >= total_cells - min_free_cells:
        print(f"Erreur: Trop d'obstacles ! Maximum: {total_cells - min_free_cells}")
        return False

    if rows < 2 or cols < 2:
        print("Erreur: La grille doit avoir au moins 2x2 cellules")
        return False

    return True


def get_user_input(prompt: str, default: int, min_val: int, max_val: int) -> int:
    """
    Demande une valeur à l'utilisateur avec validation

    Args:
        prompt: Message à afficher
        default: Valeur par défaut
        min_val: Valeur minimale
        max_val: Valeur maximale

    Returns:
        Valeur saisie par l'utilisateur
    """
    while True:
        try:
            user_input = input(f"{prompt} (défaut: {default}, min: {min_val}, max: {max_val}): ").strip()

            if user_input == "":
                return default

            value = int(user_input)

            if min_val <= value <= max_val:
                return value
            else:
                print(f"Valeur invalide ! Doit être entre {min_val} et {max_val}")

        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier")


def setup_environment_interactive() -> dict:
    """
    Configuration interactive de l'environnement

    Returns:
        Dictionnaire avec les paramètres de configuration
    """
    print("\n" + "="*60)
    print("CONFIGURATION DE L'ENVIRONNEMENT GRIDWORLD")
    print("="*60)
    print("Appuyez sur Entrée pour utiliser la valeur par défaut\n")

    # Dimension de la grille
    print("--- DIMENSIONS DE LA GRILLE ---")
    rows = get_user_input("Nombre de lignes", 6, 2, 20)
    cols = get_user_input("Nombre de colonnes", 6, 2, 20)

    total_cells = rows * cols
    max_obstacles = total_cells - 3  # Au moins 3 cellules libres

    # Nombre d'obstacles
    print("\n--- OBSTACLES ---")
    print(f"Cellules totales: {total_cells}")
    print(f"Maximum d'obstacles possibles: {max_obstacles}")
    num_obstacles = get_user_input("Nombre d'obstacles", min(6, max_obstacles), 0, max_obstacles)

    # Vérifier la validité
    while not is_valid_configuration(rows, cols, num_obstacles):
        num_obstacles = get_user_input("Nombre d'obstacles", min(6, max_obstacles), 0, max_obstacles)

    # Position de départ
    print("\n--- POSITION DE DÉPART ---")
    print("Par défaut: coin supérieur gauche (0, 0)")
    use_default_start = input("Utiliser la position par défaut ? (o/n, défaut: o): ").strip().lower()

    if use_default_start in ['n', 'non']:
        start_row = get_user_input("Ligne de départ", 0, 0, rows - 1)
        start_col = get_user_input("Colonne de départ", 0, 0, cols - 1)
        start_pos = (start_row, start_col)
    else:
        start_pos = (0, 0)

    # Position du goal
    print("\n--- POSITION DU GOAL ---")
    print(f"Par défaut: coin inférieur droit ({rows - 1}, {cols - 1})")
    use_default_goal = input("Utiliser la position par défaut ? (o/n, défaut: o): ").strip().lower()

    if use_default_goal in ['n', 'non']:
        goal_row = get_user_input("Ligne du goal", rows - 1, 0, rows - 1)
        goal_col = get_user_input("Colonne du goal", cols - 1, 0, cols - 1)
        goal_pos = (goal_row, goal_col)
    else:
        goal_pos = (rows - 1, cols - 1)

    # Vérifier que start != goal
    while start_pos == goal_pos:
        print("Erreur: La position de départ et du goal ne peuvent pas être identiques !")
        goal_row = get_user_input("Ligne du goal", rows - 1, 0, rows - 1)
        goal_col = get_user_input("Colonne du goal", cols - 1, 0, cols - 1)
        goal_pos = (goal_row, goal_col)

    # Générer les obstacles
    print("\n--- GÉNÉRATION DES OBSTACLES ---")
    obstacles = generate_random_obstacles(rows, cols, num_obstacles, start_pos, goal_pos)
    print(f"✓ {len(obstacles)} obstacles générés")

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DE LA CONFIGURATION")
    print("="*60)
    print(f"Grille: {rows} x {cols} ({total_cells} cellules)")
    print(f"Position de départ: {start_pos}")
    print(f"Position du goal: {goal_pos}")
    print(f"Nombre d'obstacles: {len(obstacles)}")
    print(f"Cellules libres: {total_cells - len(obstacles)}")
    print("="*60)

    confirm = input("\nConfirmer cette configuration ? (o/n, défaut: o): ").strip().lower()
    if confirm in ['n', 'non']:
        print("Configuration annulée. Utilisation de la configuration par défaut.")
        return None

    return {
        'grid_size': (rows, cols),
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'obstacles': obstacles
    }


def setup_environment_quick() -> dict:
    """
    Configuration rapide avec quelques presets

    Returns:
        Dictionnaire avec les paramètres de configuration
    """
    print("\n" + "="*60)
    print("CONFIGURATION RAPIDE - PRESETS")
    print("="*60)
    print("1. Petit (5x5, 3 obstacles) - Facile")
    print("2. Moyen (8x8, 8 obstacles) - Intermédiaire")
    print("3. Grand (10x10, 15 obstacles) - Difficile")
    print("4. Très grand (15x15, 30 obstacles) - Très difficile")
    print("5. Configuration personnalisée")
    print("="*60)

    choice = input("Votre choix (1-5, défaut: 2): ").strip()

    presets = {
        '1': {'grid_size': (5, 5), 'num_obstacles': 3},
        '2': {'grid_size': (8, 8), 'num_obstacles': 8},
        '3': {'grid_size': (10, 10), 'num_obstacles': 15},
        '4': {'grid_size': (15, 15), 'num_obstacles': 30},
    }

    if choice in presets:
        preset = presets[choice]
        rows, cols = preset['grid_size']
        start_pos = (0, 0)
        goal_pos = (rows - 1, cols - 1)
        obstacles = generate_random_obstacles(rows, cols, preset['num_obstacles'],
                                              start_pos, goal_pos)

        print(f"\n✓ Configuration: Grille {rows}x{cols}, {len(obstacles)} obstacles")

        return {
            'grid_size': (rows, cols),
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'obstacles': obstacles
        }
    else:
        return setup_environment_interactive()
