import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from skimage.measure import block_reduce
from morans import morans
np.seterr(all="raise")


def matrix_autocorr(s_flatten,lag):
    epsilon = 10**(-10)
    sbar = np.mean(s_flatten,axis=0)
    s_full = s_flatten - np.tile(sbar,(s_flatten.shape[0],1))
    
    s_0 = s_full[:s_flatten.shape[0]-lag,:]
    s_shift = s_full[lag:s_flatten.shape[0],:]
    
    ac_num = np.nansum(np.multiply(s_0,s_shift),axis=0)
    ac_denom = (np.nansum(np.multiply(s_full,s_full),axis=0)+epsilon)
    
    ac = np.divide(ac_num,ac_denom)
    
    return ac



def temporal_ews(s,t_roll_window):
    s = s.reshape(s.shape[0],-1)
    
    t_var = np.zeros(s.shape)
    t_skew = np.zeros(s.shape)
    t_kurt = np.zeros(s.shape)
    t_corr_1 = np.zeros(s.shape)
    t_corr_2 = np.zeros(s.shape)
    t_corr_3 = np.zeros(s.shape)
    
    for j in range(s.shape[0]-t_roll_window):
        window_end = j+t_roll_window
        s_window = s[j:window_end,:]
        
        t_var[window_end,:] = np.nanvar(s_window,axis=0)
        t_skew[window_end,:] = skew(s_window,axis=0,nan_policy='omit')
        t_kurt[window_end,:] = kurtosis(s_window,axis=0,nan_policy='omit')
        
        t_corr_1[window_end,:] = matrix_autocorr(s_window,1)
        t_corr_2[window_end,:] = matrix_autocorr(s_window,2)
        t_corr_3[window_end,:] = matrix_autocorr(s_window,3)

    
    return {'t_var':t_var,
            't_skew':t_skew,
            't_kurt':t_kurt,
            't_corr_1':t_corr_1,
            't_corr_2':t_corr_2,
            't_corr_3':t_corr_3}



def compute_ews(s,t_roll_window):
    
    # Temporal EWS:
    #t_roll_window = int(np.floor(t_roll_window_frac*s.shape[0]))
    
    t_ews = temporal_ews(s,t_roll_window)
    
   

    t_var = np.nanmean(t_ews['t_var'],axis=1)
    t_skew = np.nanmean(t_ews['t_skew'],axis=1)
    t_kurt = np.nanmean(t_ews['t_kurt'],axis=1)
    t_corr_1 = np.nanmean(t_ews['t_corr_1'],axis=1)
    t_corr_2 = np.nanmean(t_ews['t_corr_2'],axis=1)
    t_corr_3 = np.nanmean(t_ews['t_corr_3'],axis=1)
    
    # Spatial EWS:
    s_flatten = s.reshape(s.shape[0],-1)
    s_flatten = s_flatten[t_roll_window:,:]
    
    x_var = np.zeros(s.shape[0])
    x_skew = np.zeros(s.shape[0])
    x_kurt = np.zeros(s.shape[0])
    x_corr_1 = np.zeros(s.shape[0])
    x_corr_2 = np.zeros(s.shape[0])
    x_corr_3 = np.zeros(s.shape[0])
    
    x_var[t_roll_window:] = np.nanvar(s_flatten,axis=1)
    x_skew[t_roll_window:] = skew(s_flatten,axis=1,nan_policy='omit')
    x_kurt[t_roll_window:] = kurtosis(s_flatten,axis=1,nan_policy='omit')

    
    x_corr_1[t_roll_window:] = morans(s[t_roll_window:,:,:],1,periodic=False)
    x_corr_2[t_roll_window:] = morans(s[t_roll_window:,:,:],2,periodic=False)
    x_corr_3[t_roll_window:] = morans(s[t_roll_window:,:,:],3,periodic=False)
    
    x = np.vstack((t_var,t_skew,t_kurt,t_corr_1,t_corr_2,t_corr_3,
                   x_var,x_skew,x_kurt,x_corr_1,x_corr_2,x_corr_3))
    
    return x.T