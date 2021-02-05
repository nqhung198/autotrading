import pandas as pd
import numpy as np
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
    return dx[d-1:]
    '''adx = np.zeros(len(dx))
    adx[d-1] = np.mean(dx[0:d])
    for l in range(d,len(dx)):
        adx[l] = (adx[l-1]*(d-1)+dx[l])/d
    adx = adx[d-1:]
    return adx'''