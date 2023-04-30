 Dans ce jeu de science-fiction amusant et excitant, vous incarnez un combattant spatial qui doit défendre la Terre contre une armée d'envahisseurs extraterrestres. Les envahisseurs viennent vers vous en ligne, essayant de vous toucher avec leurs rayons laser, et votre mission est de les détruire tous avant qu'ils n'atteignent le sol. Vous contrôlez votre vaisseau spatial avec les flèches de direction de votre clavier ou de votre manette de jeu et appuyez sur la touche "espace" pour tirer sur les envahisseurs.
 
 ## Abstract:
 Nous avons conçu un jeu Invaders Space, avec une technique de Deep Q-learning avec un algorithme e-greedy.
 

#STEP 0: INSTALLATION DE PACKAGES

!pip install tensorflow==2.12.0 gym keras-rl2 gym[atari]

Cette partie consiste dans un premier temps à une installation tensorflow, keras et gym.
Ces packages, notamment gym nous permettra de créer notre environnement de jeu c'est-à-dire celui de Invaders Space.
Pour tensorflow et keras, ils nous permettent de créer notre agent et sa politique pour l'algorithme de Q-learning

#STEP 1: CREATION ENVIRONNEMENT AVEC OpenAI Gym

import gym 
import random
env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n
env.unwrapped.get_action_meanings()

Dans cette seconde partie de notre projet, nous créons l'environnement de notre jeu ainsi que notre agent. Au niveau de notre agent, nous pouvons le voir évoluer 
dans l'environnement avec ces différentes positions de déplacement.
Ces positions sont: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

Graçe à ce code, nous pouvons constater les différentes positions et actions de notre agent afin d'arrêter les invahisseurs pour ne pas arriver vers le bas.
Il peut aller à gauche, à droite, au centre, centre gauche et centre droite. 
A l'exécution de notre environnement, à chaque épisode, on a un scrore et au final un score définitif.

STEP 3: LA CREATION D'UN AGENT: Deep Q-Learning

 	Step 3-1: Création du modèle du deep


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers.legacy import Adam

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
		
model = build_model(height, width, channels, actions)
model.summary()

from rl.agents import DQNAgent  # Une importation de l'agent
from rl.memory import SequentialMemory ## La mémoire 
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy ## Une policy 

Dans cette étape, nous importons notre tensorflow ainsi des fonctions.
Ici nous avons créer une fonction, qui nous permet créer un modèle incluant les réseaux de neuronnes convolutifs pour notre jeu.
Nous avons un réseau de neuronnes convolutifs avec 32, 64, 512 et 256 neuronnes


Step 3-2: Création de l'agent avec le Q-learning



def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000
                  )
    return dqn
		
dqn = build_agent(model, actions) 
dqn.compile(Adam(learning_rate=0.0004)) ## La compilation de notre agent issu d'une polity EpsGreedyQPolicy

dqn.fit(env, nb_steps=10000, visualize=False, verbose=2) 

Dans cette étape nous créons un agent, sa policy et une mémoire pour jouer au jeu.
Nous avons l'entraînement de notre jeu qui prend assez de temps dû à l'interface graphique

Step 3-3: Le test du jeu Space Invaders

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

Dans cette sous étape, nous avons le test de notre jeu avec les différents scores.
Après par le code ci-dessous nous sauvegardons notre jeu et le chargeons.


