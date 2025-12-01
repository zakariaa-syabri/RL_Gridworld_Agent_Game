# GridWorld Q-Learning - Apprentissage par Renforcement

Projet de **Reinforcement Learning** avec un environnement GridWorld interactif. L'agent utilise **Q-Learning** pour apprendre par **expérience** (essais/erreurs) et non par planification.

##  Caractéristiques principales

- ✅ **Apprentissage RÉEL** : Q-Learning au lieu de Value Iteration (l'agent apprend vraiment)
- ✅ **Visualisation en temps réel** : Regardez l'agent apprendre épisode par épisode
- ✅ **Configuration flexible** : Presets ou configuration personnalisée complète
- ✅ **Goal dynamique** : Le goal change à chaque épisode pour tester l'adaptabilité
- ✅ **Génération aléatoire** : Obstacles générés automatiquement
- ✅ **Courbes d'apprentissage** : Visualisez la progression de l'agent
- ✅ **Interface interactive** : Menu simple et intuitif

##  Structure du Projet

```
RL_game/
├── gridworld_env.py           # Environnement GridWorld (inspiré de Gymnasium)
├── q_learning_agent.py        # Agent Q-Learning (apprentissage réel)
├── value_iteration_agent.py   # Agents Random et Value Iteration (optionnel)
├── environment_setup.py       # Configuration interactive et presets
├── config.py                  # Configuration par défaut
├── main.py                    # Programme principal
├── README.md                  # Ce fichier
└── GUIDE_UTILISATION.md       # Guide détaillé
```

##  Installation

### Prérequis
- Python 3.7+
- NumPy
- Matplotlib

### Installation des dépendances

```bash
pip install numpy matplotlib
```

##  Utilisation

### Démarrage rapide

```bash
python main.py
```

### Menu de configuration

Au démarrage, 3 options s'offrent à vous :

#### 1️⃣ Configuration par défaut
- Grille : 6x6
- Obstacles : 6 (prédéfinis)
- Démarrage immédiat

#### 2️⃣ Presets rapides
- **Petit** (5x5, 3 obstacles) → Facile, apprentissage rapide
- **Moyen** (8x8, 8 obstacles) → Intermédiaire, équilibré
- **Grand** (10x10, 15 obstacles) → Difficile, plus de complexité
- **Très grand** (15x15, 30 obstacles) → Très difficile, maximum challenge

#### 3️⃣ Configuration personnalisée
Choisissez :
- Dimensions de la grille (2-20 lignes/colonnes)
- Nombre d'obstacles (avec validation automatique)
- Position de départ
- Position du goal
- Génération aléatoire des obstacles

## Ce que vous allez voir

### 1. Test de l'agent random (baseline)
Performance de base avec actions aléatoires

### 2. Entraînement Q-Learning visualisé

**Tous les 50 épisodes**, vous verrez :
- L'agent se déplacer dans la grille
- Les Q-values évoluer en temps réel
- Mode EXPLORATION → EXPLOITATION
- Epsilon décroître (100% → 1%)
- Trajectoires s'améliorer

**Progression typique** :
```
Épisode 50  : Exploration pure (mouvements aléatoires)
Épisode 100 : Commence à apprendre des patterns
Épisode 200 : Trajectoires plus efficaces
Épisode 300 : Performance quasi-optimale
```

### 3. Courbes d'apprentissage

4 graphiques montrant :
- **Récompenses** : Évolution des récompenses par épisode
- **Steps** : Nombre de pas par épisode (décroît)
- **Epsilon** : Décroissance de l'exploration
- **Taux de succès** : Pourcentage de réussite

### 4. Animation avec goal dynamique

L'agent s'adapte à de nouveaux goals :
- Goal change à chaque épisode
- L'agent recalcule rapidement
- Démontre la généralisation

### 5. Visualisation finale

Grille avec :
- **Q-values** : Valeurs optimales de chaque état
- **Politique** : Flèches indiquant les meilleures actions
- **Couleurs** : Intensité selon la valeur
- **Agent** : Cercle rouge se déplaçant

##  Configuration

### Fichier `config.py`

```python
# Dimensions (si mode par défaut)
GRID_SIZE = (6, 6)
START_POS = (0, 0)
GOAL_POS = (5, 5)
OBSTACLES = [(1,1), (1,2), (2,3), (3,3), (4,1), (4,2)]

# Q-Learning
USE_Q_LEARNING = True          # Active Q-Learning
LEARNING_RATE = 0.1            # Vitesse d'apprentissage
EPSILON_START = 1.0            # Exploration initiale (100%)
EPSILON_DECAY = 0.995          # Décroissance d'epsilon
EPSILON_MIN = 0.01             # Exploration minimale (1%)
NUM_TRAINING_EPISODES = 300    # Nombre d'épisodes d'entraînement

# Visualisation de l'entraînement
VISUALIZE_TRAINING = True      # Voir l'agent apprendre
VISUALIZE_EVERY = 50           # Visualiser tous les N épisodes
TRAINING_ANIMATION_DELAY = 0.1 # Vitesse de l'animation
SHOW_Q_VALUES_TRAINING = True  # Afficher les Q-values

# Goal dynamique
DYNAMIC_GOAL = True            # Goal change à chaque épisode
NUM_ANIMATED_EPISODES = 3      # Nombre d'épisodes à animer
```

### Ajuster selon la taille de la grille

**Petite grille (5x5)** :
```python
NUM_TRAINING_EPISODES = 150
VISUALIZE_EVERY = 30
```

**Moyenne grille (8x8)** :
```python
NUM_TRAINING_EPISODES = 300
VISUALIZE_EVERY = 50
```

**Grande grille (10x10+)** :
```python
NUM_TRAINING_EPISODES = 500
VISUALIZE_EVERY = 100
```

##  Algorithme Q-Learning

Q-Learning est un algorithme d'**apprentissage par renforcement** sans modèle (model-free).

### Règle de mise à jour

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Où :
- **Q(s,a)** : Valeur Q de l'état s avec l'action a
- **α** : Learning rate (vitesse d'apprentissage)
- **r** : Récompense immédiate
- **γ** : Gamma (discount factor)
- **s'** : État suivant
- **max_a' Q(s',a')** : Meilleure Q-value du prochain état

### Exploration vs Exploitation (ε-greedy)

L'agent équilibre :
- **Exploration** : Essayer de nouvelles actions (ε = epsilon)
- **Exploitation** : Utiliser les meilleures actions connues (1-ε)

```
Début : ε = 100% → Exploration pure
Fin   : ε = 1%   → Exploitation quasi-pure
```

##  Exemple de sortie

```
=======================================================================
     GRIDWORLD - REINFORCEMENT LEARNING avec Q-Learning
=======================================================================

📋 MODE DE CONFIGURATION
-----------------------------------------------------------------------
1. Configuration par défaut (fichier config.py)
2. Presets rapides (Petit/Moyen/Grand/Très grand)
3. Configuration personnalisée (Interactive)
-----------------------------------------------------------------------
Votre choix (1-3, défaut: 1): 2

=======================================================================
CONFIGURATION RAPIDE - PRESETS
=======================================================================
1. Petit (5x5, 3 obstacles) - Facile
2. Moyen (8x8, 8 obstacles) - Intermédiaire
3. Grand (10x10, 15 obstacles) - Difficile
4. Très grand (15x15, 30 obstacles) - Très difficile
5. Configuration personnalisée
=======================================================================
Votre choix (1-5, défaut: 2): 2

✓ Configuration: Grille 8x8, 8 obstacles

=======================================================================
ENVIRONNEMENT CRÉÉ
=======================================================================
Taille de la grille: (8, 8)
Position de départ: (0, 0)
Position du goal: (7, 7)
Nombre d'obstacles: 8
Cellules libres: 56
=======================================================================

==================================================
ENTRAÎNEMENT PAR Q-LEARNING (APPRENTISSAGE RÉEL)
==================================================
L'agent va APPRENDRE par essais/erreurs

Paramètres:
  - Learning rate (alpha): 0.1
  - Gamma (discount): 0.95
  - Epsilon (exploration): 1.0 → 0.01
  - Nombre d'épisodes: 300

Début de l'entraînement sur 300 épisodes...
Visualisation activée tous les 50 épisodes

>>> Visualisation de l'épisode 50...
  [Animation de l'agent explorant la grille]

Épisode 50/300 | Récompense moy: -0.450 | Steps moy: 45.2 | Succès: 12.0% | Epsilon: 0.778

>>> Visualisation de l'épisode 100...
  [Animation avec trajectoires plus efficaces]

Épisode 100/300 | Récompense moy: 0.234 | Steps moy: 28.5 | Succès: 58.0% | Epsilon: 0.605

>>> Visualisation de l'épisode 150...
  [Animation avec trajectoires quasi-optimales]

Épisode 150/300 | Récompense moy: 0.678 | Steps moy: 15.2 | Succès: 88.0% | Epsilon: 0.471

...

Entraînement terminé!

==================================================
COURBES D'APPRENTISSAGE
==================================================
[Affichage des 4 graphiques de progression]

==================================================
TEST DE L'AGENT Q-LEARNING ENTRAÎNÉ
==================================================
Test en mode exploitation (epsilon = 0)...

Épisode 1: Récompense = 0.891, Steps = 12 ✓
Épisode 2: Récompense = 0.891, Steps = 12 ✓
Épisode 3: Récompense = 0.891, Steps = 12 ✓
Épisode 4: Récompense = 0.891, Steps = 12 ✓
Épisode 5: Récompense = 0.891, Steps = 12 ✓

Récompense moyenne: 0.891
Taux de succès: 100.0%

==================================================
ANIMATION DES ÉPISODES
==================================================
Animation de 3 épisodes avec GOAL DYNAMIQUE...
Le goal change à chaque épisode pour tester l'adaptabilité de l'agent.

Épisode animé 1/3...
  Nouveau goal: (3, 7)
  Réentraînement pour le nouveau goal...
  [L'agent s'adapte au nouveau goal]
  → Terminé: 8 steps, récompense = 0.921 ✓

...
```

##  Fonctionnalités avancées

### Goal dynamique

Avec `DYNAMIC_GOAL = True`, le goal change à chaque épisode :
- Teste la **généralisation** de l'agent
- Prouve que l'agent comprend la **structure** du gridworld
- Pas juste mémorisation, mais **vraie compréhension**

### Génération aléatoire d'obstacles

Les obstacles sont générés automatiquement :
- Distribution aléatoire dans la grille
- Évite automatiquement start et goal
- Garantit un chemin possible

### Validation automatique

Le système valide :
- Grille minimale : 2x2
- Au moins 3 cellules libres (start + goal + chemin)
- Start ≠ Goal
- Nombre d'obstacles valide

##  Fichiers principaux

### `gridworld_env.py`
Environnement GridWorld (inspiré de Gymnasium) :
- Méthodes : `reset()`, `step()`, `get_next_state()`, etc.
- 4 actions : UP, DOWN, LEFT, RIGHT
- Système de récompenses
- Support goal dynamique

### `q_learning_agent.py`
Agent Q-Learning :
- Apprentissage par expérience
- Exploration epsilon-greedy
- Mise à jour des Q-values
- Extraction de la politique optimale

### `environment_setup.py`
Configuration interactive :
- Presets rapides
- Configuration personnalisée
- Génération d'obstacles
- Validation

### `main.py`
Programme principal :
- Menu de configuration
- Entraînement visualisé
- Courbes d'apprentissage
- Animations

##  Conseils d'utilisation

### Pour débuter
1. Utilisez le preset **Petit** (option 2 → 1)
2. Observez l'entraînement tous les 30 épisodes
3. Notez comment les trajectoires s'améliorent

### Pour expérimenter
1. Créez une configuration personnalisée (option 3)
2. Testez différentes tailles et densités d'obstacles
3. Ajustez `NUM_TRAINING_EPISODES` selon la complexité

### Pour comprendre Q-Learning
1. Regardez les Q-values évoluer pendant l'entraînement
2. Observez epsilon décroître (exploration → exploitation)
3. Comparez les courbes de récompenses et steps

##  Roncontre des problématiques

**L'agent n'apprend pas** :
- Augmenter `NUM_TRAINING_EPISODES`
- Vérifier qu'il y a un chemin vers le goal
- Ajuster `LEARNING_RATE` (essayer 0.05 ou 0.2)

**Visualisation trop lente** :
- Augmenter `VISUALIZE_EVERY` (ex: 100)
- Réduire `TRAINING_ANIMATION_DELAY` (ex: 0.05)
- Désactiver `VISUALIZE_TRAINING` temporairement

**Grille trop grande** :
- Augmenter `NUM_TRAINING_EPISODES` proportionnellement
- Pour 15x15 : au moins 500 épisodes recommandés


## 📄 Licence

Projet éducatif de démonstration pour l'apprentissage du Reinforcement Learning.

##  Auteur

Projet créé pour illustrer les concepts de Reinforcement Learning avec Q-Learning.
SYABRI Zakariaa
---


