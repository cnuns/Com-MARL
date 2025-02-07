import torch
import numpy as np

STATE = {'Good': 1, 'Bad': 0}

def model_error_statistics(p, r, k, h):
    """
    Determine error statistics given model parameters.

    Determines the proportion of time spent in the Bad state, the error rate, 
    and the expected burst length of error patterns generated with the given 
    Gilbert-Elliot burst error model parameters. In the case of the two-parameter
    model, where k=1 and h=0, the lag one correlation and relative expected
    burst error length are also included in the error_stats dict.
    
    Parameters
    ----------
    p : float
        Probability of transitioning from the Good state to the Bad state.
    r : float
        Probability of transitioning from the Bad state to the Good state.
    k : float
        Probability of no error occurring when in the Good state.
    h : float
        Probability of no error occurring when in the Bad state.

    Returns
    -------
    error_stats : dict
        Dictionary for relevant error statistics
    """

    bad_proportion = p/(p + r)
    error_rate = ((1 - k) * r + (1 - h) * p)/(p + r)
    expected_burst_length = ((1 - k) * r + (1 - h) * p)\
        /((1 - k) * r * ((1 - p) * k + p*h) + (1 - h) * p * ((1 - r) * h + r*k))
    error_stats = {'bad_proportion': bad_proportion,
                   'error_rate': error_rate,
                   'expected_burst_length': expected_burst_length,
                   }
    if k == 1 and h == 0:
        # In two-parameter model we will calculate lag-one correlation and
        # relative expected burst length
        lag_one_correlation = 1 - r - p
        relative_expected_burst_length = (1 - error_rate)/r

        error_stats['lag_one_correlation'] = lag_one_correlation
        error_stats['relative_expected_burst_length'] = relative_expected_burst_length

    return error_stats

def get_GE_num_mean_state_changes(Pgb, Pbg, n=1000, init_state=STATE['Good'], Tmax=25):
    switches = []
    for i in range(n):
        states = [init_state]
        for i in range(Tmax-1):
            next_state = get_next_state(states[-1], Pgb, Pbg)
            states.append(next_state)

        # states_ = list(map(str, states))
        # out_state = " ".join(states_)

        output = np.array(states)
        real_bad_ratio = sum(output == STATE['Bad']) / Tmax
        # print(statistics['bad_proportion'], real_bad_ratio)
        # print('#'*50)
        
        switch_cnt = 0
        for i, e in enumerate(output):
            if i == 0:
                bef = e
                continue

            if bef != e: # state changed
                switch_cnt += 1
            else:
                pass
            bef = e
        # print(f'state switch count: {switch_cnt}')
        switches.append(switch_cnt)
    
    return round(sum(switches)/len(switches), 1)

def get_init_state(n, Pgb, Pbg):
    bad_rate = Pgb/(Pgb+Pbg)
    state = torch.rand(size=(n,n)) >= bad_rate # if a number greater than bad_rate, then set state as (Good=True=1)
    return state
    

def event_occur(p_threshold):
    rand = torch.rand(size=(1,1))
    if rand >= p_threshold:
        return False
    else:
        return True
        
def get_next_state(state, Pgb, Pbg):
    """
        Args:
            Pgb: Probability of From Good state to go to Bad state, "p" in Figure 1
            Pbg: Probability of From Bad state to go to Good state, "r" in Figure 1
    """
    if state == STATE['Good']:
        # We`re in State Good`

        # Determine our next state
        if event_occur(p_threshold=Pgb):
            next_state = STATE['Bad']
        else:
            next_state = STATE['Good']

    else:
        if event_occur(p_threshold=Pbg):
            next_state = STATE['Good']
        else:
            next_state = STATE['Bad']

    return next_state


def get_next_state_matrix(n_sequence:int, state:torch.Tensor, Pgb:float, Pbg:float, include_prev=False):
    """
        good: 1
        bad: 0
    """
    if type(state) != torch.Tensor:
        state = torch.from_numpy(state).bool()

    n = state.shape[-1]

    if include_prev:
        state_seq = [state]
    else:
        state_seq = []

    for i in range(n_sequence):
        #! good link state update
        event_good_to_bad = (torch.rand(size=(n,n)) + torch.eye(n)) < Pgb
        G_next_state = state * ~(state * event_good_to_bad)
        
        #! bad link state update
        event_bad_to_good = (torch.rand(size=(n,n)) + torch.eye(n)) < Pbg
        B_next_state = ~state * event_bad_to_good
        
        next_state = G_next_state + B_next_state
        state_seq.append(next_state)
        state = next_state

    state_seq = torch.stack(state_seq)
    return state_seq

def calc_burst_length(brst:torch.Tensor, state:torch.Tensor):
    brst = brst + (~state)
    brst = brst * ~state
    return brst

def calc_bad_count(bad_count:torch.Tensor, state:torch.Tensor):
    bad_count = bad_count + (~state)
    return bad_count

def calc_state_switch_count(switch:torch.Tensor, old_state:torch.Tensor, state:torch.Tensor):
    switch = switch + torch.logical_xor(old_state, state)
    return switch


if __name__ == '__main__':
    # Apply GE_loss at the single-value level.
    p = Pgb = 0.1
    r = Pbg = 0.2
    p = Pgb = 0.0196
    r = Pbg = 0.282
    
    statistics = model_error_statistics(p, r, k=1, h=0)
    print(statistics)

    bad_rate = p/(p + r)
    init_state = STATE['Bad'] if event_occur(bad_rate)==True else STATE['Good']
    states = [init_state]
    states = [STATE['Good']]

    seq_length = 400
    for i in range(seq_length):
        next_state = get_next_state(states[-1], Pgb, Pbg)
        states.append(next_state)

    states_ = list(map(str, states))
    out_state = " ".join(states_)

    output = np.array(states)
    
    real_bad_ratio = sum(output == STATE['Bad']) / seq_length
    print(statistics['bad_proportion'], real_bad_ratio)
    print('#'*50)
    
    switch_cnt = 0
    for i, e in enumerate(output):
        if i == 0:
            bef = e
            continue

        if bef != e: # state changed
            switch_cnt += 1
        else:
            pass
        bef = e
    print(f'state switch count: {switch_cnt}')