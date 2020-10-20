import numpy as np

def sample_const_delay(d):
    return (lambda: d)

item_difficulty_mean = 1
item_difficulty_std = 1

log_item_decay_exp_mean = 1
log_item_decay_exp_std = 1

log_delay_coef_mean = 0
log_delay_coef_std = 0.01

def sample_item_difficulties(n_items):
    return np.random.normal(item_difficulty_mean, item_difficulty_std, n_items)

def sample_student_ability():
    return 0

def sample_window_cw(n_windows):
    x = 1 / (np.arange(1, n_windows+1, 1))**2
    return x[::-1]
  
def sample_window_nw(n_windows):
    x = 1 / (np.arange(1, n_windows+1, 1))**2
    return x[::-1]

def sample_item_decay_exps(n_items):
    return np.exp(np.random.normal(log_item_decay_exp_mean, log_item_decay_exp_std, n_items))

def sample_student_decay_exp():
    return 0

def sample_delay_coef():
    return np.exp(np.random.normal(log_delay_coef_mean, log_delay_coef_std))

def sample_item_decay_rates(n_items):
    return np.exp(np.random.normal(np.log(0.077), 1, n_items))

def sample_loglinear_coeffs(n_items):
    coeffs = np.array([1, 1, 0])
    coeffs = np.concatenate((coeffs, np.random.normal(0, 1, n_items)))
    return coeffs
