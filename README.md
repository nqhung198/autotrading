# Autotrading based Reinforcement Learning



This repository provides the code for a Reinforcement Learning trading agent with its trading environment that works with both simulated and historical market data. 

This repository has the Keras implementation of
- [Deep Q-Network (DQN)](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
- [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461?source=post_page-----c0de4471f368----------------------)
- [Dueling Double DQN (DDDQN)](https://arxiv.org/abs/1511.06581)

[Code for agents](/Agent/)


![gif](https://github.com/nqhung198/autotrading/blob/master/fe800gif.gif)


### Requirements
- Python 3.5/3.6
- Keras 
- Tensorflow 


### Features
- 3 Reinforcement learning Agents (DQN, DDQN, DDDQN)
- ADX and RSI technical indicator and extensible for more
- Historical stock market data ingestion through CSV

## Dataset
### Raw data
Raw data is data we put in files by filesname para.
Columns:
  - open (Open Price): price at begin in (a minute)
  - high (High Price): price at highest point while (a minute)
  - low (Low Price): price at lowest point while (a minute)
  - close (Close Price): price at the end of minute
  - volume: Volume trade of market. 
Every row is timestep of updating prices from market.
Up to your trading strategy we choses the timeframe,
Specialy i chose minutes, which mean one minute per 
timestep

(picture)

### Generate data
Transformed data from raw data to put into the agent
The function _genarate will transform and yield step by 
step, see below.

(picture)

In dataframe, it's mean..

(picture)

ask, bid: easy to know
mid: avarage ask and bid
rsi, cci, adx: the indicators (_14 is default period)

## Policy
### State (Input data and shape)
[adx, rsi, cci, price, 𝑢𝑛𝑟𝑒𝑎𝑙𝑖𝑧𝑒𝑑 𝑟𝑒𝑡𝑢𝑟𝑛, 𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛]

adx, rsi, cci: indicators
price: the oject price close of market
𝑢𝑛𝑟𝑒𝑎𝑙𝑖𝑧𝑒𝑑 𝑟𝑒𝑡𝑢𝑟𝑛: 
  if long unrealized_return = (price_(t-1) - price_(t)) / price_(t-1)
  if short unrealized_return = (price_(t) - price_(t-1)) / price_(t-1)
𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛: ('flat': [1, 0, 0],
          'long': [0, 1, 0],
          'short':[0, 0, 1])

example: 
[-0.2464357 -0.20249262 -0.25567938 -0.94463668  0.00208518  0. 0. 1.]

adx: -0.2464357
rsi: -0.20249262
cci: -0.25567938
price: -0.94463668
𝑢𝑛𝑟𝑒𝑎𝑙𝑖𝑧𝑒𝑑 𝑟𝑒𝑡𝑢𝑟𝑛: 0.00208518
𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛: [0. 0. 1.]

### Action
The agent could take three actions – Buy, Sell or Hold

### Reward
The reward objective is set to maximize realized PnL from a round trip trade. 
Which mean: 
- Every timestep: Reward <- Reward - time_fee(e.g 0.001)  (penalty)
- Every trade/order/invest:
    Reward <- Reward - trade_fee(e.g 0.005)  (commision penalty)
    If Profit:
      + Reward <- Reward + Profit
    If Loss:
      + Reward <- Reward + Loss (Loss<0)
Reward update every state.



### What's next?
- Prioritized Experience Replay
- LSTM networks
- Asynchronous Advantage Actor-Critic (A3C)
- Multi-agent
- Reward engineering
