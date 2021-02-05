import pandas as pd
import os
import random
import numpy as np
import time
import re
import MetaTrader5 as mt5
from Environment.gens.TA_Gen import TAStreamer
from Environment.envs.indicator_1 import Indicator_1

from Agent.duelling_dqn import DDDQNAgent

def World(filename=None,
        train_test = 'train',
        episodes=10,
        train_test_split = 0.75,
        trading_fee = .0002,
        time_fee = .001,
        memory_size = 3000,
        gamma = 0.99,
        epsilon_min = 0.01,
        batch_size = 64,
        train_interval = 10,
        learning_rate = 0.001,
        render_show=False,
        display=False,
        save_results=False
):
    start = time.time()

    generator = TAStreamer(filename=filename, mode='train', split=train_test_split)
    episode_length = round(int(len(pd.read_csv(filename))*train_test_split), -1)

    environment = Indicator_1(data_generator=generator,
                              trading_fee=trading_fee,
                              time_fee=time_fee,
                              episode_length=episode_length)
    action_size = len(Indicator_1._actions)

    state = environment.reset()

    state_size = len(state)


    try:
        symbol = re.findall(r'C:\\Users\\ndhun\\Autotrding\\([^.]+)',filename)[0]
    except:
        symbol = ""

    agent = DDDQNAgent(state_size=state_size,
                     action_size=action_size,
                     memory_size=memory_size,
                     episodes=episodes,
                     episode_length=episode_length,
                     train_interval=train_interval,
                     gamma=gamma,
                     learning_rate=learning_rate,
                     batch_size=batch_size,
                     epsilon_min=epsilon_min,
                     train_test=train_test,
                     symbol=symbol)

    # Warming up the agent
    if (train_test == 'train'):
        for _ in range(memory_size):
            action = agent.act(state)
            next_state, reward, done, _ = environment.step(action)
            agent.observe(state, action, reward, next_state, done, warming_up=True)
        if display:
            print('completed mem allocation: ', time.time() - start)

    # Training the agent
    loss_list=[]
    val_loss_list=[]
    reward_list=[]
    epsilon_list=[]
    metrics_df=None
    if train_test == "train":
        best_loss = 9999
        best_reward = 0
        for ep in range(episodes):
            ms = time.time()
            state = environment.reset()
            rew = 0
            loss_list_temp = []
            val_loss_list_temp = []

            for _ in range(episode_length):
                action = agent.act(state)
                next_state, reward, done, _ = environment.step(action)
                loss = agent.observe(state, action, reward, next_state,
                                     done)  # loss would be none if the episode length is not % by 10
                state = next_state
                rew += reward
                if(loss):
                    loss_list_temp.append(round(loss.history["loss"][0],3))
                    val_loss_list_temp.append(round(loss.history["val_loss"][0],3))

            if display:
                print("Ep:" + str(ep)
                      + "| rew:" + str(round(rew, 2))
                      + "| eps:" + str(round(agent.epsilon, 2))
                      + "| loss:" + str(round(loss.history["loss"][0], 4))
                      + "| runtime:" + str(time.time() - ms))
                print("Loss=", str(np.mean(loss_list_temp)), " Val_Loss=", str(np.mean(val_loss_list_temp)))

            loss_list.append(np.mean(loss_list_temp))
            val_loss_list.append(np.mean(val_loss_list_temp))
            reward_list.append(rew)
            epsilon_list.append(round(agent.epsilon, 2))

        agent.save_model()

        metrics_df=pd.DataFrame({'loss':loss_list,'val_loss':val_loss_list,'reward':reward_list,'epsilon':epsilon_list})

        if save_results:
            metrics_df.to_csv(r'C:\Users\ndhun\Autotrding\perf_metrics.csv')

    if(train_test=='test'):
        agent.load_model()

        generator = TAStreamer(filename=filename, mode='test', split=train_test_split)
        environment = Indicator_1(data_generator=generator,
                              trading_fee=trading_fee,
                              time_fee=time_fee,
                              episode_length=episode_length,)
    if train_test=='trade':
        agent.load_model()
        generator = TAStreamer(filename=filename, mode='trade',period=4, split=train_test_split)
        environment = Indicator_1(data_generator=generator,
                            trading_fee=trading_fee,
                            time_fee=time_fee,
                            episode_length=episode_length)
        if not mt5.initialize():
            print("initialize() failed, error code =",mt5.last_error())
            quit()
        print(mt5.version())
        # connect to the trade account without specifying a password and a server
        # attempt to enable the display of the GBPUSD in MarketWatch
        selected=mt5.symbol_select("GBPUSD",True)
        if not selected:
            print("Failed to select GBPUSD")
            mt5.shutdown()
            quit()
        # prepare the buy request structure
        symbol = "GBPUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(symbol, "not found, can not call order_check()")
            mt5.shutdown()
            quit()
        account=50481525
        authorized=mt5.login(account)  # the terminal database password is applied if connection data is set to be remembered
        if authorized:
            print("connected to account #{}".format(account))
            print(mt5.account_info())
        else:
            print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
 
        # if the symbol is unavailable in MarketWatch, add it
        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(symbol,True):
                print("symbol_select({}}) failed, exit",symbol)
                mt5.shutdown()
                quit()
        
    done = False
    state = environment.reset()
    q_values_list=[]
    state_list=[]
    action_list=[]
    reward_list=[]
    trade_list=[]

    while not done:
        action, q_values = agent.act(state, test=True)
        state, reward, done, info = environment.step(action,True)
        if 'status' in info and info['status'] == 'Closed plot':
            done = True
        else:
            reward_list.append(reward)

            calc_returns=environment.return_calc(render_show)
            if calc_returns:
                trade_list.append(calc_returns)

            if(render_show):
                environment.render()

        if train_test=='trade':
            print('Waiting..')
        else:
            #if bought --> tpsl 
            q_values_list.append(q_values)
            state_list.append(state)
            action_list.append(action)

            print('Reward = %.2f' % sum(reward_list))

            trades_df=pd.DataFrame(trade_list)
            action_policy_df = pd.DataFrame({'q_values':q_values_list,'state':state_list,'action':action_list})

            if save_results:
                trades_df.to_csv(r'C:\Users\ndhun\Autotrding\trade_list.csv')
                action_policy_df.to_pickle(r'C:\Users\ndhun\Autotrding\action_policy.pkl')
                action_policy_df.to_csv(r'C:\Users\ndhun\Autotrding\action_policy_CSV.csv')
            if display:
                print("All done:", str(time.time() - start))

    return({"metrics_df":metrics_df,
            "trades_df":trades_df,
            "action_policy_df":action_policy_df,
            "reward_list":reward_list})


if __name__ == "__main__":
    
    World(filename = 'C:\\Users\\ndhun\\Autotrding\\gbpusd1.csv', episodes=1, display=True,  train_test='trade')
    # World()
    #World(train_test='trade'), environment.step(action, True)
