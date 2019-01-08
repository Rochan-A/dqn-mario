# Reinforcement Learning for Super Mario Bros using DQN
## Modified to work with [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

> NOTE:
This is an unofficial fork of the original code published as [dqn-mario](https://github.com/nailo2c/dqn-mario).

# DQN

使用PyTorch實作DQN演算法，並訓練super-mario-bros以及atari-pong，整體架構參考openai/baselines。

*Warning*：訓練DQN請開足夠的記憶體，Replay Buffer以預設值1000000為例至少會使用約8G的記憶體.
# Dependencies

* Python 3.6
* PyTorch
* gym
* gym-super-mario-bros

# Result

* Super-Mario-Bros

使用8顆cpu在GCP上跑16個小時，RAM開24G非常足夠，但很難收斂，無法穩定過關。
訓練的影像預設位置在/video/mario/。

![](img/mario-dqn-16hr.gif)

## NOTE: Atari Pong section of the code has not been tested. I have left it commented out.

# References

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[openai/baselines](https://github.com/openai/baselines)  
[transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)  
[openai/gym](https://github.com/openai/gym)  
[Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
