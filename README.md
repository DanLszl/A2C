# A2C
An implementation of `Synchronous Advantage Actor Critic (A2C)` in TensorFlow. A2C is a variant of advantage actor critic introduced by [OpenAI in their published baselines](https://github.com/openai/baselines). However, these baselines are difficult to understand and modify. So, I implemented the A2C based on their implementation but in a clearer and simpler way.


## Asynchronous vs Synchronous Advantage Actor Critic
Asynchronous advantage actor critic was introduced in [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf). The difference between both methods is that in asynchronous AC, parallel agents update the global network each one on its own. So, at a certain time, the weights used by an agent maybe different than the weights used by another agent leading to the fact that each agent plays with a different policy to explore more and more of the environment. However, in synchronous AC, all of the updates by the parallel agents are collected to update the global network. To encourage exploration, stochastic noise is added to the probability distribution of the actions predicted by each agent.

### Environments Supported
This implementation allows for using different environments. It's not restricted to OpenAI gym environments. If you want to attach the project to another environment rather than that provided by gym, all you have to do is to inherit from the base class `BaseEnv` in `envs/base_env.py`, and implement all the methods in a plug and play fashion (See the gym environment example class). You also have to add the name of the new environment class in `A2C.py\env_name_parser()` method.

The methods that should be implemented in the new environment class are: 
1. `make()` for creating the environment and returning a reference to it.
2. `step()` for taking a step in the environment and returning a tuple (observation images, reward float value, done boolean, any other info).
3. `reset()` for resetting the environment to the initial state.
4. `get_observation_space()` for returning an object with attribute tuple `shape` representing the shape of the observation space.
5. `get_action_space()` for returing an object with attribute tuple `n` representing the number of possible actions in the environment.
6. `render()` for rendering the environment if appropriate.

### Policy Models Supported
This implementation comes with the basic CNN policy network from OpenAI baseline. However, it supports using different policy networks. All you have to do is to inherit from the base class `BasePolicy` in `models\base_policy.py`, and implement all the methods in a plug and play fashion again :D (See the CNNPolicy example class).

### Tensorboard Visualization
This implementation allows for the beautiful Tensorboard visualization. It displays the time plots per running agent of the two most important signals in reinforcement learning: episode length and total reward in the episode. All you have to do is to launch Tensorboard from your experiment directory located in `experiments/`.
```
tensorboard --logdir=experiments/my_experiment/summaries
```
### Video Producing
During training, you can generate videos of the trained agent playing the game. This is achieved by changing `record_video_every` in the configuration file from -1 to the number of episodes between two generated videos. Generated videos are in your experiment directory.

During testing, videos are generated automatically if the optional `monitor` method is implemented in the environment.

## Usage
### Main Dependencies
 ```
 Python 3 or above
 tensorflow 1.3.0
 numpy 1.13.1
 gym 0.9.2
 tqdm 4.15.0
 bunch 1.0.1
 matplotlib 2.0.2
 Pillow 4.2.1
 ```
### Run
```
python main.py config/test.json
```
The file 'test.json' is just an example of a file having all parameters to train on environments. You can create your own configuration file for training/testing.

In the project, two configuration files are provided as examples for training on Pong and Breakout Atari games.

## Results
|   Model   |   Game   | Average Score | Max Score |
|:---------:|:--------:|:-------------:|:---------:|
| CNNPolicy |   Pong   |       17      |     21    |
| CNNPolicy | Breakout |      650      |    850    |

## Updates
* Inference and training are working properly.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.