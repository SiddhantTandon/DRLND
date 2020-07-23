<center><h1>Udacity - DRLND</h1></center>
<center><h2>Project 1: Navigation</h2></center>
<center><h3>Report by Siddhant Tandon</h3></center>
<center><h4>Date: 18 May 2020</h4></center>

### Environment
The project uses an environment built in Unity. The environment contains a brain which is responsible for deciding the actions of their associated agents. Below are the characteristics of the environment.

- Unity brain name: BananaBrain
- Vector Observation space type: continuous
- Vector Observation space size (per agent): 37
- Number of stacked Vector Observation: 1
- Vector Action space type: discrete
- Vector Action space size (per agent): 4

### Learning Algorithm

In this project a Deep Q-Network learning algorithm is used with the following steps:

1. Sample environment and store experience tuples **(s, a, r, s')** and store it in replay buffer with some size
2. Once the experience buffer has enough tuples and certain threshold is crossed - get random minibatch from experience buffer and train the network on those tuples
3. While training, fix the target weights for learning steps to make the algoritm more stable and update set of weights using gradient descent.

### Overview of DQN

The implementation is utilizing Deep Q-Network with 3 hidden layers (500, 200, 100 nodes respectively).
Hyperparameters chosen for the implementation are below:

```python
BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
SEED = 42
EPISODES = 2000
MAX_TIMESTEPS = int(1e3)
hidden_layers = [500, 200, 100]
```

### Results

DQn algorithm was able to reach the environment's reward threshold in 498 episodes.

```
Episode 100	Average Score: 0.07
Episode 200	Average Score: 1.92
Episode 300	Average Score: 6.56
Episode 400	Average Score: 10.69
Episode 492	Average Score: 13.02
Environment solved in 392 episodes!	Average Score: 13.02
```
[DQN](/images/navigation_plot.jpg)
The weights of the networks are stored in `saved_weights.pth` file using `torch.save(agent.local.state_dict(), 'saved_weights.pth')`

### Ideas for Future Work

The Deep Q-Network can be further improved by for example:

- Modifying replay buffer to make the experience tuples prioritized: the higher the TD error the higher priority
- Using Double DQN - one set of weights for selecting best action and other one to evaluate it
- Using Dueling DQN (DDQN) - Introduce so called advantage values **A(s, a)** - the advantage of taking the action at the state (how much better is it to take this action vs other). Then use one network to estimate state values and the other that estimates advantages **A(s, a)**. From both of those comptre Q values.
