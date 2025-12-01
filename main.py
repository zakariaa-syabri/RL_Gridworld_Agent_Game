"""
Programme principal pour le GridWorld avec Reinforcement Learning
Affiche la grille, les value states et la politique optimale
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
from gridworld_env import GridWorldEnv
from value_iteration_agent import ValueIterationAgent, RandomAgent
from q_learning_agent import QLearningAgent
from environment_setup import setup_environment_interactive, setup_environment_quick
import config
import time


def visualize_gridworld(env: GridWorldEnv, agent: ValueIterationAgent = None,
                        show_values: bool = True, show_policy: bool = True):
    """
    Visualise le GridWorld avec matplotlib

    Args:
        env: Environnement GridWorld
        agent: Agent Value Iteration (optionnel)
        show_values: Afficher les valeurs des états
        show_policy: Afficher la politique
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Obtenir les valeurs et la politique
    if agent:
        values = agent.get_value_grid()
        policy = agent.get_policy_grid()
    else:
        values = np.zeros((env.rows, env.cols))
        policy = {}

    # Normaliser les valeurs pour la colormap
    v_min = values.min()
    v_max = values.max()

    # Dessiner chaque cellule
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)

            # Déterminer la couleur de la cellule
            if state == env.start_pos:
                color = 'lightgreen'
                edge_color = 'darkgreen'
                edge_width = 3
            elif state == env.goal_pos:
                color = 'gold'
                edge_color = 'orange'
                edge_width = 3
            elif state in env.obstacles:
                color = 'black'
                edge_color = 'black'
                edge_width = 1
            else:
                # Couleur basée sur la valeur
                if v_max != v_min:
                    normalized_value = (values[row, col] - v_min) / (v_max - v_min)
                else:
                    normalized_value = 0
                color = plt.cm.Blues(normalized_value * 0.7 + 0.3)
                edge_color = 'gray'
                edge_width = 1

            # Dessiner la cellule
            rect = FancyBboxPatch((col, env.rows - row - 1), 1, 1,
                                   boxstyle="round,pad=0.05",
                                   linewidth=edge_width,
                                   edgecolor=edge_color,
                                   facecolor=color)
            ax.add_patch(rect)

            # Afficher les valeurs
            if show_values and state not in env.obstacles:
                value_text = f'{values[row, col]:.3f}'
                ax.text(col + 0.5, env.rows - row - 0.65,
                       value_text,
                       ha='center', va='center',
                       fontsize=config.FONT_SIZE_VALUE,
                       color='black' if state not in [env.start_pos, env.goal_pos] else 'darkblue',
                       weight='bold')

            # Afficher la politique
            if show_policy and state in policy:
                policy_symbol = policy[state]
                ax.text(col + 0.5, env.rows - row - 0.3,
                       policy_symbol,
                       ha='center', va='center',
                       fontsize=config.FONT_SIZE_POLICY,
                       color='darkred',
                       weight='bold')

            # Ajouter des labels pour start et goal
            if state == env.start_pos:
                ax.text(col + 0.5, env.rows - row - 0.15,
                       'START',
                       ha='center', va='center',
                       fontsize=8,
                       color='darkgreen',
                       weight='bold')
            elif state == env.goal_pos:
                ax.text(col + 0.5, env.rows - row - 0.15,
                       'GOAL',
                       ha='center', va='center',
                       fontsize=8,
                       color='darkorange',
                       weight='bold')

    # Configuration des axes
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Colonnes', fontsize=12)
    ax.set_ylabel('Lignes', fontsize=12)
    ax.set_title('GridWorld - Value Iteration\n(Valeurs des états et politique optimale)',
                fontsize=14, weight='bold', pad=20)

    # Inverser l'axe y pour que (0,0) soit en haut à gauche
    ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax


def plot_learning_curves(agent):
    """
    Affiche les courbes d'apprentissage

    Args:
        agent: Agent Q-Learning
    """
    if not isinstance(agent, QLearningAgent) or not agent.training_complete:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Récompenses par épisode
    axes[0, 0].plot(agent.episode_rewards, alpha=0.3, color='blue', label='Récompense')
    # Moyenne mobile
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(agent.episode_rewards)), moving_avg,
                       color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    axes[0, 0].set_xlabel('Épisode')
    axes[0, 0].set_ylabel('Récompense')
    axes[0, 0].set_title('Évolution des récompenses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Steps par épisode
    axes[0, 1].plot(agent.episode_steps, alpha=0.3, color='green', label='Steps')
    if len(agent.episode_steps) >= window:
        moving_avg_steps = np.convolve(agent.episode_steps, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(agent.episode_steps)), moving_avg_steps,
                       color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    axes[0, 1].set_xlabel('Épisode')
    axes[0, 1].set_ylabel('Nombre de steps')
    axes[0, 1].set_title('Évolution du nombre de steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Epsilon (exploration)
    axes[1, 0].plot(agent.epsilon_history, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Épisode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Décroissance de l\'exploration (epsilon)')
    axes[1, 0].grid(True, alpha=0.3)

    # Taux de succès cumulé
    successes = [1 if r > 0 else 0 for r in agent.episode_rewards]
    cumulative_success = []
    for i in range(len(successes)):
        if i < window:
            cumulative_success.append(np.mean(successes[:i+1]) * 100)
        else:
            cumulative_success.append(np.mean(successes[i-window+1:i+1]) * 100)
    axes[1, 1].plot(cumulative_success, color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Épisode')
    axes[1, 1].set_ylabel('Taux de succès (%)')
    axes[1, 1].set_title(f'Taux de succès (fenêtre de {window} épisodes)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 105])

    plt.tight_layout()
    return fig


def draw_grid_base(ax, env: GridWorldEnv, agent=None,
                   show_values: bool = True, show_policy: bool = True):
    """
    Dessine la grille de base sans l'agent

    Args:
        ax: Axes matplotlib
        env: Environnement GridWorld
        agent: Agent (ValueIteration ou QLearning)
        show_values: Afficher les valeurs des états
        show_policy: Afficher la politique
    """
    ax.clear()

    # Obtenir les valeurs et la politique (supporte ValueIteration et QLearning)
    if agent and hasattr(agent, 'get_value_grid'):
        values = agent.get_value_grid()
        policy = agent.get_policy_grid()
    else:
        values = np.zeros((env.rows, env.cols))
        policy = {}

    # Normaliser les valeurs pour la colormap
    v_min = values.min()
    v_max = values.max()

    # Dessiner chaque cellule
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)

            # Déterminer la couleur de la cellule
            if state == env.start_pos:
                color = 'lightgreen'
                edge_color = 'darkgreen'
                edge_width = 3
            elif state == env.goal_pos:
                color = 'gold'
                edge_color = 'orange'
                edge_width = 3
            elif state in env.obstacles:
                color = 'black'
                edge_color = 'black'
                edge_width = 1
            else:
                # Couleur basée sur la valeur
                if v_max != v_min:
                    normalized_value = (values[row, col] - v_min) / (v_max - v_min)
                else:
                    normalized_value = 0
                color = plt.cm.Blues(normalized_value * 0.7 + 0.3)
                edge_color = 'gray'
                edge_width = 1

            # Dessiner la cellule
            rect = FancyBboxPatch((col, env.rows - row - 1), 1, 1,
                                   boxstyle="round,pad=0.05",
                                   linewidth=edge_width,
                                   edgecolor=edge_color,
                                   facecolor=color)
            ax.add_patch(rect)

            # Afficher les valeurs
            if show_values and state not in env.obstacles:
                value_text = f'{values[row, col]:.2f}'
                ax.text(col + 0.5, env.rows - row - 0.7,
                       value_text,
                       ha='center', va='center',
                       fontsize=config.FONT_SIZE_VALUE - 1,
                       color='black' if state not in [env.start_pos, env.goal_pos] else 'darkblue',
                       weight='bold')

            # Afficher la politique
            if show_policy and state in policy:
                policy_symbol = policy[state]
                ax.text(col + 0.5, env.rows - row - 0.3,
                       policy_symbol,
                       ha='center', va='center',
                       fontsize=config.FONT_SIZE_POLICY,
                       color='darkred',
                       weight='bold')

            # Ajouter des labels pour start et goal
            if state == env.start_pos:
                ax.text(col + 0.5, env.rows - row - 0.15,
                       'START',
                       ha='center', va='center',
                       fontsize=7,
                       color='darkgreen',
                       weight='bold')
            elif state == env.goal_pos:
                ax.text(col + 0.5, env.rows - row - 0.15,
                       'GOAL',
                       ha='center', va='center',
                       fontsize=7,
                       color='darkorange',
                       weight='bold')

    # Configuration des axes
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Colonnes', fontsize=12)
    ax.set_ylabel('Lignes', fontsize=12)
    ax.invert_yaxis()


def visualize_training_episode(env: GridWorldEnv, agent: QLearningAgent,
                               episode_num: int, total_episodes: int,
                               max_steps: int = 100):
    """
    Visualise un épisode d'entraînement en temps réel

    Args:
        env: Environnement GridWorld
        agent: Agent Q-Learning
        episode_num: Numéro de l'épisode
        total_episodes: Nombre total d'épisodes
        max_steps: Nombre maximum de steps

    Returns:
        (récompense totale, nombre de steps)
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Réinitialiser l'environnement
    state = env.reset()
    total_reward = 0
    steps = 0
    trajectory = [state]

    while not env.done and steps < max_steps:
        # Dessiner la grille avec Q-values
        draw_grid_base(ax, env, agent,
                      show_values=config.SHOW_Q_VALUES_TRAINING,
                      show_policy=False)

        # Dessiner l'agent
        agent_row, agent_col = state
        agent_circle = Circle((agent_col + 0.5, env.rows - agent_row - 0.5),
                             0.35, color='red', zorder=10, linewidth=2,
                             edgecolor='darkred')
        ax.add_patch(agent_circle)

        # Dessiner la trajectoire
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                r1, c1 = trajectory[i]
                r2, c2 = trajectory[i + 1]
                ax.plot([c1 + 0.5, c2 + 0.5],
                       [env.rows - r1 - 0.5, env.rows - r2 - 0.5],
                       'r-', alpha=0.3, linewidth=2)

        # Titre avec informations détaillées
        mode = "EXPLORATION" if np.random.random() < agent.epsilon else "EXPLOITATION"
        title = f'ENTRAÎNEMENT - Épisode {episode_num}/{total_episodes}\n'
        title += f'Step: {steps + 1} | Position: {state} | Goal: {env.goal_pos}\n'
        title += f'Récompense cumulée: {total_reward:.3f} | Epsilon: {agent.epsilon:.3f} ({mode})'
        ax.set_title(title, fontsize=13, weight='bold', pad=20)

        plt.draw()
        plt.pause(config.TRAINING_ANIMATION_DELAY)

        # Sélectionner une action (avec exploration)
        action = agent.select_action(state, training=True)

        # Exécuter l'action
        next_state, reward, done, info = env.step(action)

        # Mettre à jour la Q-value
        agent.update_q_value(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        trajectory.append(state)
        steps += 1

    # Afficher l'état final
    draw_grid_base(ax, env, agent,
                  show_values=config.SHOW_Q_VALUES_TRAINING,
                  show_policy=False)

    agent_row, agent_col = state
    agent_circle = Circle((agent_col + 0.5, env.rows - agent_row - 0.5),
                         0.35, color='red', zorder=10, linewidth=2,
                         edgecolor='darkred')
    ax.add_patch(agent_circle)

    # Dessiner la trajectoire complète
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            r1, c1 = trajectory[i]
            r2, c2 = trajectory[i + 1]
            ax.plot([c1 + 0.5, c2 + 0.5],
                   [env.rows - r1 - 0.5, env.rows - r2 - 0.5],
                   'r-', alpha=0.5, linewidth=2)

    success = "✓ GOAL ATTEINT!" if env.done else "✗ Échec"
    title = f'ENTRAÎNEMENT - Épisode {episode_num}/{total_episodes} - {success}\n'
    title += f'Steps: {steps} | Récompense totale: {total_reward:.3f} | Epsilon: {agent.epsilon:.3f}'
    ax.set_title(title, fontsize=13, weight='bold', pad=20)

    plt.draw()
    plt.pause(1.0)
    plt.ioff()
    plt.close(fig)

    # Décroissance d'epsilon
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    return total_reward, steps


def animate_episode(env: GridWorldEnv, agent, episode_num: int,
                    show_values: bool = True, show_policy: bool = True,
                    dynamic_goal: bool = False):
    """
    Anime un épisode en montrant l'agent se déplacer

    Args:
        env: Environnement GridWorld
        agent: Agent à utiliser
        episode_num: Numéro de l'épisode
        show_values: Afficher les valeurs des états
        show_policy: Afficher la politique
        dynamic_goal: Si True, change le goal et recalcule les valeurs
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Changer le goal si dynamique
    if dynamic_goal:
        new_goal = env.get_random_goal()
        print(f"  Nouveau goal: {new_goal}")
        env.set_goal(new_goal)

        # Adapter l'agent au nouveau goal
        if isinstance(agent, ValueIterationAgent):
            # Value Iteration : recalcul complet
            print(f"  Recalcul des valeurs...", end=" ")
            iterations = agent.value_iteration(max_iterations=config.MAX_ITERATIONS)
            if config.SHOW_VALUE_CONVERGENCE:
                print(f"convergé en {iterations} itérations")
            else:
                print("OK")
        elif isinstance(agent, QLearningAgent):
            # Q-Learning : réentraînement rapide avec expérience conservée
            print(f"  Réentraînement pour le nouveau goal...")
            agent.reset_for_new_goal()
            agent.train(num_episodes=100, max_steps=config.MAX_STEPS_PER_EPISODE,
                       verbose_every=50)

    # Réinitialiser l'environnement
    state = env.reset()
    episode_reward = 0
    steps = 0
    trajectory = [state]

    # Exécuter l'épisode
    while not env.done and steps < config.MAX_STEPS_PER_EPISODE:
        # Dessiner la grille
        draw_grid_base(ax, env, agent, show_values, show_policy)

        # Dessiner l'agent
        agent_row, agent_col = state
        agent_circle = Circle((agent_col + 0.5, env.rows - agent_row - 0.5),
                             0.3, color='red', zorder=10)
        ax.add_patch(agent_circle)

        # Titre avec informations
        title = f'Épisode {episode_num} - Step {steps + 1}\n'
        title += f'Position: {state} | Goal: {env.goal_pos} | Récompense cumulée: {episode_reward:.3f}'
        ax.set_title(title, fontsize=14, weight='bold', pad=20)

        plt.draw()
        plt.pause(config.ANIMATION_DELAY)

        # Sélectionner et exécuter une action
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        trajectory.append(state)
        steps += 1

    # Afficher l'état final
    draw_grid_base(ax, env, agent, show_values, show_policy)

    agent_row, agent_col = state
    agent_circle = Circle((agent_col + 0.5, env.rows - agent_row - 0.5),
                         0.3, color='red', zorder=10)
    ax.add_patch(agent_circle)

    success = "✓ GOAL ATTEINT!" if env.done else "✗ Échec"
    title = f'Épisode {episode_num} - TERMINÉ {success}\n'
    title += f'Position finale: {state} | Goal: {env.goal_pos} | Steps: {steps} | Récompense totale: {episode_reward:.3f}'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    plt.draw()
    plt.pause(1.5)
    plt.ioff()
    plt.close(fig)

    return episode_reward, steps, trajectory


def test_random_agent(env: GridWorldEnv, num_episodes: int = 5):
    """
    Teste l'agent random

    Args:
        env: Environnement GridWorld
        num_episodes: Nombre d'épisodes de test
    """
    print("\n" + "="*50)
    print("TEST DE L'AGENT RANDOM")
    print("="*50)

    agent = RandomAgent(env)
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while not env.done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1

        total_rewards.append(episode_reward)
        print(f"Épisode {episode + 1}: Récompense = {episode_reward:.3f}, Steps = {steps}")

    print(f"\nRécompense moyenne: {np.mean(total_rewards):.3f}")
    print(f"Récompense écart-type: {np.std(total_rewards):.3f}")


def test_value_iteration_agent(env: GridWorldEnv, agent: ValueIterationAgent, num_episodes: int = 5):
    """
    Teste l'agent Value Iteration

    Args:
        env: Environnement GridWorld
        agent: Agent Value Iteration
        num_episodes: Nombre d'épisodes de test
    """
    print("\n" + "="*50)
    print("TEST DE L'AGENT VALUE ITERATION")
    print("="*50)

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        trajectory = [state]

        while not env.done and steps < config.MAX_STEPS_PER_EPISODE:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            trajectory.append(state)
            steps += 1

        total_rewards.append(episode_reward)
        success = "✓" if env.done else "✗"
        print(f"Épisode {episode + 1}: Récompense = {episode_reward:.3f}, Steps = {steps} {success}")
        print(f"  Trajectoire: {' → '.join([str(s) for s in trajectory[:10]])}" +
              (" ..." if len(trajectory) > 10 else ""))

    print(f"\nRécompense moyenne: {np.mean(total_rewards):.3f}")
    print(f"Récompense écart-type: {np.std(total_rewards):.3f}")
    print(f"Taux de succès: {sum([1 for r in total_rewards if r > 0]) / len(total_rewards) * 100:.1f}%")


def main():
    """
    Fonction principale
    """
    print("="*70)
    print("     GRIDWORLD - REINFORCEMENT LEARNING avec Q-Learning")
    print("="*70)

    # Menu de configuration
    print("\n📋 MODE DE CONFIGURATION")
    print("-" * 70)
    print("1. Configuration par défaut (fichier config.py)")
    print("2. Presets rapides (Petit/Moyen/Grand/Très grand)")
    print("3. Configuration personnalisée (Interactive)")
    print("-" * 70)

    choice = input("Votre choix (1-3, défaut: 1): ").strip()

    # Choisir la configuration
    if choice == '2':
        env_config = setup_environment_quick()
        if env_config is None:
            env_config = {
                'grid_size': config.GRID_SIZE,
                'start_pos': config.START_POS,
                'goal_pos': config.GOAL_POS,
                'obstacles': config.OBSTACLES
            }
    elif choice == '3':
        env_config = setup_environment_interactive()
        if env_config is None:
            env_config = {
                'grid_size': config.GRID_SIZE,
                'start_pos': config.START_POS,
                'goal_pos': config.GOAL_POS,
                'obstacles': config.OBSTACLES
            }
    else:
        # Configuration par défaut
        env_config = {
            'grid_size': config.GRID_SIZE,
            'start_pos': config.START_POS,
            'goal_pos': config.GOAL_POS,
            'obstacles': config.OBSTACLES
        }
        print("\n✓ Utilisation de la configuration par défaut")

    # Créer l'environnement
    env = GridWorldEnv(
        grid_size=env_config['grid_size'],
        start_pos=env_config['start_pos'],
        goal_pos=env_config['goal_pos'],
        obstacles=env_config['obstacles']
    )

    print("\n" + "="*70)
    print("ENVIRONNEMENT CRÉÉ")
    print("="*70)
    print(f"Taille de la grille: {env_config['grid_size']}")
    print(f"Position de départ: {env_config['start_pos']}")
    print(f"Position du goal: {env_config['goal_pos']}")
    print(f"Nombre d'obstacles: {len(env_config['obstacles'])}")
    print(f"Cellules libres: {env.rows * env.cols - len(env_config['obstacles'])}")
    print("="*70)

    # Tester l'agent random
    test_random_agent(env, num_episodes=config.NUM_TEST_EPISODES)

    # Choisir l'algorithme
    if config.USE_Q_LEARNING:
        # Q-LEARNING : APPRENTISSAGE RÉEL PAR EXPÉRIENCE
        print("\n" + "="*50)
        print("ENTRAÎNEMENT PAR Q-LEARNING (APPRENTISSAGE RÉEL)")
        print("="*50)
        print("L'agent va APPRENDRE par essais/erreurs\n")

        agent = QLearningAgent(
            env,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            epsilon=config.EPSILON_START,
            epsilon_decay=config.EPSILON_DECAY,
            epsilon_min=config.EPSILON_MIN
        )

        print(f"Paramètres:")
        print(f"  - Learning rate (alpha): {config.LEARNING_RATE}")
        print(f"  - Gamma (discount): {config.GAMMA}")
        print(f"  - Epsilon (exploration): {config.EPSILON_START} → {config.EPSILON_MIN}")
        print(f"  - Nombre d'épisodes: {config.NUM_TRAINING_EPISODES}\n")

        # Entraînement avec visualisation
        if config.VISUALIZE_TRAINING:
            print(f"Mode VISUALISATION activé - vous verrez l'agent apprendre en temps réel !")
            stats = agent.train(
                num_episodes=config.NUM_TRAINING_EPISODES,
                max_steps=config.MAX_STEPS_PER_EPISODE,
                verbose_every=config.VERBOSE_EVERY,
                visualize_callback=visualize_training_episode,
                visualize_every=config.VISUALIZE_EVERY
            )
        else:
            stats = agent.train(
                num_episodes=config.NUM_TRAINING_EPISODES,
                max_steps=config.MAX_STEPS_PER_EPISODE,
                verbose_every=config.VERBOSE_EVERY
            )

        # Afficher les courbes d'apprentissage
        print("\n" + "="*50)
        print("COURBES D'APPRENTISSAGE")
        print("="*50)
        print("Affichage des courbes d'apprentissage...")
        plot_learning_curves(agent)
        plt.show()

        # Tester l'agent entraîné
        print("\n" + "="*50)
        print("TEST DE L'AGENT Q-LEARNING ENTRAÎNÉ")
        print("="*50)
        print("Test en mode exploitation (epsilon = 0)...\n")

        # Sauvegarder epsilon et le mettre à 0 pour le test
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

        total_rewards = []
        for episode in range(config.NUM_TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            steps = 0
            trajectory = [state]

            while not env.done and steps < config.MAX_STEPS_PER_EPISODE:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                state = next_state
                trajectory.append(state)
                steps += 1

            total_rewards.append(episode_reward)
            success = "✓" if env.done else "✗"
            print(f"Épisode {episode + 1}: Récompense = {episode_reward:.3f}, Steps = {steps} {success}")

        print(f"\nRécompense moyenne: {np.mean(total_rewards):.3f}")
        print(f"Taux de succès: {sum([1 for r in total_rewards if r > 0]) / len(total_rewards) * 100:.1f}%")

        # Restaurer epsilon
        agent.epsilon = old_epsilon

    else:
        # VALUE ITERATION : PLANIFICATION (ANCIEN)
        print("\n" + "="*50)
        print("ENTRAÎNEMENT PAR VALUE ITERATION (PLANIFICATION)")
        print("="*50)

        agent = ValueIterationAgent(env, gamma=config.GAMMA, theta=config.THETA)
        print(f"Gamma (discount factor): {config.GAMMA}")
        print(f"Theta (convergence threshold): {config.THETA}")
        print("\nExécution de Value Iteration...")

        iterations = agent.value_iteration(max_iterations=config.MAX_ITERATIONS)

        print(f"Convergence atteinte après {iterations} itérations")

        # Afficher quelques valeurs
        print("\nValeurs des états (échantillon):")
        print(f"  Start {config.START_POS}: {agent.V[config.START_POS]:.4f}")
        print(f"  Goal {config.GOAL_POS}: {agent.V[config.GOAL_POS]:.4f}")

        # Tester l'agent Value Iteration
        test_value_iteration_agent(env, agent, num_episodes=config.NUM_TEST_EPISODES)

    # Animer des épisodes
    print("\n" + "="*50)
    print("ANIMATION DES ÉPISODES")
    print("="*50)
    if config.DYNAMIC_GOAL:
        print(f"Animation de {config.NUM_ANIMATED_EPISODES} épisodes avec GOAL DYNAMIQUE...")
        print("Le goal change à chaque épisode pour tester l'adaptabilité de l'agent.")
    else:
        print(f"Animation de {config.NUM_ANIMATED_EPISODES} épisodes...")
    print("Fermer la fenêtre pour passer à l'épisode suivant.\n")

    for i in range(config.NUM_ANIMATED_EPISODES):
        print(f"Épisode animé {i + 1}/{config.NUM_ANIMATED_EPISODES}...")
        reward, steps, trajectory = animate_episode(env, agent, i + 1,
                                                    show_values=True,
                                                    show_policy=True,
                                                    dynamic_goal=config.DYNAMIC_GOAL)
        success = "✓" if reward > 0 else "✗"
        print(f"  → Terminé: {steps} steps, récompense = {reward:.3f} {success}\n")

    # Visualiser le GridWorld final (statique)
    print("\n" + "="*50)
    print("VISUALISATION FINALE")
    print("="*50)
    print("Affichage de la grille avec les valeurs et la politique...")

    visualize_gridworld(env, agent, show_values=True, show_policy=True)
    plt.show()

    print("\nTerminé!")


if __name__ == "__main__":
    main()
