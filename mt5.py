
import MetaTrader5 as mt5
import time
import numpy as np
from dddqn import Env, Indicator_1, Agent, DDDQNAgent
import pandas as pd
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
episodes=10
train_test_split = 0.75
trading_fee = .0002
time_fee = .0005
memory_size = 3000
gamma = 0.9
epsilon_min = 0.005
batch_size = 64
train_interval = 10
learning_rate = 0.001
render_show=False
display=False
save_results=False

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
def momentum(xc, k):
    """
        Computes momentum indicator: m_t(k) = xc_t - xc_{t - k}.

        Params:
            xc -> A pd.Series obj representing xc_t.
            k -> Time window lag

        Output:
            A pd.Series obj representing m_t(k). 
    """
    return  xc - xc.shift(k)
def RSI(xc, q = 14):
    """
    Computes Relative Strength Index:
        RS_t(q) = \frac{\sum_{i = 0}^{q - 1} m_{t - i}(1)|_{m_{t - i}(1) > 0}}
                       {-\sum_{i = 0}^{q - 1} m_{t - i}(1)|_{m_{t - i}(1) < 0}}
        where m_t(k) is the momentum indicator.

        rsi_t(q) = RS_t(q) / (1 + RS_t(q))

    Params:
        xc -> A pd.Series obj representing xc_t
        q -> Time window lag
    """
    momentum1_index = momentum(xc, 1)
    rsi_index = np.zeros(len(momentum1_index))
    rsi_index[:q - 1] = np.nan
    for i in range(q - 1, len(momentum1_index)):
        cum_increase = np.sum([x for x in momentum1_index[i - q + 1: i + 1] if x > 0])
        cum_decrease = -np.sum([x for x in momentum1_index[i - q + 1: i + 1] if x < 0])
       
        if cum_decrease == 0 and cum_increase == 0:
            rsi_index[i] = 0.5
        elif cum_decrease == 0:
            rsi_index[i] = 1
        else:
            RS = cum_increase / cum_decrease
            rsi_index[i] = RS / (1 + RS)
        
    return pd.Series(rsi_index) 
def CCI(high,low,close,period):
    tp = (np.array(high)+np.array(low)+np.array(close))/3 # typical price
    atp = np.zeros(len(high)) # average typical price
    md = np.zeros(len(high)) # mean deviation
    result = np.zeros(len(high))
    for i in range(period-1,len(high)):
        atp[i] = np.sum(tp[i-(period-1):i+1])/period
        md[i] = np.sum(np.fabs(atp[i]-tp[i-(period-1):i+1]))/period
        result[i] = (tp[i]-atp[i])/(0.015*md[i])
    return result[period-1:]
def adx(a,b,c,d):
    tr = np.zeros(len(a))
    hph = np.zeros(len(a))
    pll = np.zeros(len(a))
    trd = np.zeros(len(a))
    pdm = np.zeros(len(a))
    ndm = np.zeros(len(a))
    pdmd = np.zeros(len(a))
    ndmd = np.zeros(len(a))
    for i in range(1,len(a)):
        hl = a[i]-b[i]
        hpc = np.fabs(a[i]-c[i-1])
        lpc = np.fabs(b[i]-c[i-1])
        tr[i] = np.amax(np.array([hl,hpc,lpc]))
        hph[i] = a[i]-a[i-1]
        pll[i] = b[i-1]-b[i]
    for j in range(1,len(a)):
        if hph[j]>pll[j]:
            if hph[j]>0:
                pdm[j]=hph[j]
        if pll[j]>hph[j]:
            if pll[j]>0:
                ndm[j]=pll[j]
    trd[d]=np.sum(tr[1:d+1])
    pdmd[d]=np.sum(pdm[1:d+1])
    ndmd[d]=np.sum(ndm[1:d+1])
    for k in range(d+1,len(a)):
        trd[k]=trd[k-1]-trd[k-1]/d+tr[k]
        pdmd[k]=pdmd[k-1]-pdmd[k-1]/d+pdm[k]
        ndmd[k]=ndmd[k-1]-ndmd[k-1]/d+ndm[k]
    trd = trd[d:]
    pdmd = pdmd[d:]
    ndmd = ndmd[d:]
    p = (pdmd/trd)*100
    n = (ndmd/trd)*100
    diff = np.fabs(p-n)
    summ = p+n
    dx = 100*(diff/summ)
    adx = np.zeros(len(dx))
    adx[d-1] = np.mean(dx[0:d])
    for l in range(d,len(dx)):
        adx[l] = (adx[l-1]*(d-1)+dx[l])/d
    adx = adx[d-1:]
    return adx
def Generator(curr="GBPUSD",period=14):
    Done=True
    rates = mt5.copy_rates_from_pos(curr, mt5.TIMEFRAME_M1, 0, 2)
    c=0
    open_list=[]
    high_list=[]
    low_list=[]
    close_list=[]
    while Done:
        lasttick=mt5.symbol_info_tick(curr)
        bid, ask = round(lasttick.bid,5), round(lasttick.ask,5)
        mid = round((bid + ask)/2,5)
        c+=1
        check = rates[1][0]
        #rates = mt5.copy_rates_from_pos("GBPUSD", mt5.TIMEFRAME_M1, 0, 1)
        while check==rates[1][0]:
            rates = mt5.copy_rates_from_pos(curr, mt5.TIMEFRAME_M1, 0, 2)
        open, high, low, close, tickvol, spread = rates[0][1], rates[0][2], rates[0][3], rates[0][4], rates[0][5], rates[0][6]
        open_list.append(open)
        high_list.append(high)
        low_list.append(low)
        close_list.append(close)        
        #if c>period:
        cci = CCI(high_list, low_list, close_list, period)
        if len(cci)==0:
            cci = np.append(cci,0)
        rsi = RSI(pd.Series(close_list), period)
        yield np.array([bid, ask, mid, round(rsi.values[-1],5), cci[-1]/100])
        #else:
        #    print('Collecting Data.. {}/{}'.format(c,period))

environment = Indicator_1(data_generator=Generator('GBPUSD',4), trading_fee=trading_fee,time_fee=time_fee)

action_size = len(Indicator_1._actions)
state_size = len(state)
try:
    symbol = 'data1000' # Model's name
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
agent.load_model()
done = False
state = environment.reset()
q_values_list=[]
state_list=[]
action_list=[]
reward_list=[]
trade_list=[]

while not done:
    action, q_values = agent.act(state, test=True)
    state, reward, done, info = environment.step(action)
    if 'status' in info and info['status'] == 'Closed plot':
        done = True
    else:
        reward_list.append(reward)

        calc_returns=environment.return_calc(render_show)
        if calc_returns:
            trade_list.append(calc_returns)

        if(render_show):
            environment.render()


    q_values_list.append(q_values)
    state_list.append(state)
    action_list.append(action)
    if action == [0,1,0]:
        #buy/tpsl sell
        #if buy --> buy
        print('Buy or TP/SL')
        #if sold --> tpsl
    elif action == [0,0,1]:
        #sell/tpsl buy
        #if sell --> sell
        print('Sell or TP/SL')

        #if bought --> tpsl 