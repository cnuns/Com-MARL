import os
import pandas as pd
import numpy as np
import pickle
from glob import glob
import csv
import pandas as pd
from glob import glob
import numpy as np
import time
import torch

from custom_implement.utils import set_policy_attributes
from torch.nn.utils.rnn import pad_sequence

def save_matlab(file_path, backup_path, epoch_data, rewMat2, mat_file,
                trAvgReward, trAvgSuccess, trainMetric,
                xlist, xEnvlist,
                params,
                vectors,
                formats='mat'):
    """
        Args:
            epoch_data: epoch-episodes-maxsteps(200) all kind of reward, size:(epoch, episodes, maxsteps)
            rewMat2: epoch-episodes reward, size:(epoch, episode)
            mat_file: avgRew, avgSuc, cntRs, cntRp, ... existing data formats
            xlist: Env.Step Evaluation Points
    """
    if formats == 'mat':
        from scipy.io import savemat
        # epoch_data['success'] = np.transpose(epoch_data['success'], (0, -1, -2))
        rewMat3 = {}
        for vec in vectors:
            tem_ = epoch_data[vec]
            tem_ = np.array(tem_)
            tem_ = np.transpose(tem_, (2, 1, 0))
            rewMat3[vec] = tem_
        rewMat3['success'] = np.transpose(epoch_data['success'], (2, 1, 0))

        # backup file save
        with open(f'{backup_path}.pkl', 'wb') as f:
            data = {'epoch_data': epoch_data, 'rewMat2':rewMat2, 'mat_file':mat_file}
            pickle.dump(data, f)
            
        data = {k:np.array(v).reshape(-1,1) for k,v in mat_file.items()}
        data['params'] = params
        data['trainRew'] = trAvgReward.astype(np.float64).reshape(-1,1) # row vector to column vector
        data['trainSuc'] = trAvgSuccess.astype(np.float64).reshape(-1,1)
        data['trainMetric'] = trainMetric.astype(np.float64).reshape(-1,1)
        data['xlist'] = xlist.astype(np.float64).reshape(-1,1)
        data['xEnvlist'] = xEnvlist.astype(np.float64).reshape(-1,1)
        
        data['rewMat3'] = rewMat3
        data['rewMat2'] = rewMat2
        savemat(f'{file_path}.mat', data)
        
    elif formats == 'json':
        import json
        data = {'data':data}
        with open(f"{file_path}.json", "w") as json_file:
            json.dump(data, json_file)
        
    elif formats == 'pkl':
        with open(f'{file_path}.pkl', 'wb') as f:
            pickle.dump(data, f)

import joblib, json
def load_policy(exp_dir, epoch):
    #* parameter load
    try:
        policy = joblib.load(f'{exp_dir}/itrs/itr_{epoch}.pkl')
    except:
        policy = joblib.load(f'{exp_dir}/itrs/itr_{str(epoch).zfill(4)}.pkl')
    with open(f"{exp_dir}/variant.json", "r") as st_json:
        params = json.load(st_json)['args_dict']

    return policy

def load_backup(backup_path):
    if os.path.isfile(f'{backup_path}.pkl'):
        with open(f'{backup_path}.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None
    
def get_train_avg_result(model_path, col_name):
    seq = {}
    csvs = glob(f'{model_path}/*.csv')
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except:
            continue
        idx = np.min(df['Iteration'].to_numpy())
        seq[idx] = df[col_name].to_numpy()

    idxs = sorted(list(seq.keys()))
    series = []
    for idx in idxs:
        series.append(seq[idx])
    train_result = np.concatenate(series, axis=0)

    return train_result

def add_train_avg_result(csvPath, model_path, train_avg_result, colName):
    df = pd.read_csv(csvPath)
    series_train = pd.Series(train_avg_result, name=colName)
    df_train = pd.DataFrame(series_train)
    new_df = pd.concat([df_train, df], axis=1)
    new_df.to_csv(csvPath, index=False)

def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for item1, item2 in zip(list1, list2):
        if item1 != item2:
            return False

    return True

def read_csv_columns(file_name):
    columns = []
    
    if os.path.exists(file_name):
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file)
            try:
                columns = next(reader)  # Get the column names
                data = list(reader)
                
                # Transpose the data to separate lists for each column
                data_columns = list(map(list, zip(*data)))
            
                return columns #, data_columns
            except:
                return None
    else:
        print(f"The file '{file_name}' does not exist.")
        return None
    
def check_csv_file_exist(csvPath, cols, end):
    isAlreadyFin = False
    existFile = False
    existCol = False
    start = None

    if os.path.isfile(csvPath):
        existFile = True

    columns = read_csv_columns(csvPath)
    if columns != None:
        if not are_lists_equal(columns, cols):
            print('This has already been tested.')
            isAlreadyFin = True
            existCol = True
        else:
            df = pd.read_csv(csvPath)
            if len(df) >= 1:
                df = df[np.isfinite(df['iter'])]
                iter_max = np.max(df['iter'].to_numpy())
                print(f'{csvPath}:\nrestore test from: {iter_max} ---> to: {end}')
                start = int(iter_max)
                existCol = True
            else:
                pass
    else:
        existCol = False
        
    return existFile, existCol, start, isAlreadyFin

def get_test_range_EnvStep_file(model_path, scale, start=1.5, end=-1, num_points=51, allTest=False):
    #_range = np.linspace(start, end, num_points)
    d = []
    for c in glob(f'{model_path}/*.csv'):
        df = pd.read_csv(c)
        idx = np.min(df['Iteration'].to_numpy())
        d.append((idx, df))
    d = sorted(d)
    sorted_dfs = [df for k,df in d]
    df = pd.concat(sorted_dfs, axis=0)

    itrs = {'epoch': df['Iteration'].to_numpy(), 'envstep': df['TotalEnvSteps'].to_numpy()/scale}

    if allTest:
        return itrs['epoch'], itrs['envstep']

    if end == -1:
        end = itrs['envstep'][-1]
        
    _range = np.linspace(start, end, num_points)

    eval_epochs = []
    for envstep in _range:
        indices = np.where(itrs['envstep']>=envstep)[0][0]
        #print(indices)
        eval_epochs.append(indices)

    if min(eval_epochs) == 0 and eval_epochs[0] == 0:
        eval_epochs = np.array(eval_epochs) + 1
    else:
        eval_epochs = np.array(eval_epochs)

    return eval_epochs, _range


TEST_METRIC_VECTORS = ['iter', 'EnvStep', 'success', 'avgRew', 'capture_cnt', 'step_cnt', 'penalty_cnt', 'variable', 'deg', 'variable', 'vars2']
VECTORS = ['reward', 'capture_cnt', 'step_cnt', 'move_cnt', 'penalty_cnt', 'nodeDeg', 'variable', 'vars2']

MATLAB_COLS = ['avgRew', 'cntRc', 'cntRm', 'cntRp', 'cntRs', 'cntRv', 'cntRv2']
MATLAB_COLS_MAPPING = {'avgRew': 'reward', 'cntRc': 'capture_cnt', 'cntRm': 'move_cnt', 'cntRp': 'penalty_cnt', 'cntRs': 'step_cnt', 'cntRv':'variable', 'cntRv2': 'vars2'}
def test_and_output_result_file(args, env, flag, model_path, path_test, scenLib, eval_model):
    time_start = time.time()
    baseColumns = TEST_METRIC_VECTORS + ['time']
    csvPath = f'{path_test}/csv/{args.exp_name}.csv'
    pkl_path = f'{args.dir_backup}/{args.exp_name}'
    mat_path = f'{args.dir_matlab}/{args.exp_name}'

    offset = 0
    epoch_list = np.array(args.epoch_list)
    cnt = 0
    

    trAvgReward = get_train_avg_result(model_path, col_name='AverageReturn')
    trAvgSuccess = get_train_avg_result(model_path, col_name='SuccessRate')
    cntRc = get_train_avg_result(model_path, col_name='AverageCaptureCount')
    cntRs = get_train_avg_result(model_path, col_name='AverageStepCount')
    cntRm = get_train_avg_result(model_path, col_name='AverageMovingCount')
    cntRp = get_train_avg_result(model_path, col_name='AveragePenaltyCount')
    cntRv = get_train_avg_result(model_path, col_name='AverageVariable')
    try:
        cntRv2 = get_train_avg_result(model_path, col_name='AverageVar2')
    except:
        cntRv2 = np.zeros_like(cntRv)
    optimal = get_train_avg_result(model_path, col_name='BoundReturn')

    trainMetric = scenLib.calc_metric(args, trAvgReward, optimal, cntRc, cntRs, cntRm, cntRp, cntRv, cntRv2)

    xEnvlist2 = np.array(args.xEnvlist[:])

    data = load_backup(pkl_path)
    if data:
        epoch_data = data['epoch_data']
        rewMat2 = data['rewMat2']
        mat_file = data['mat_file']
    else:
        mat_file = {c:[] for c in MATLAB_COLS}
        mat_file['optRew'] = []
        mat_file['avgSucc'] = []

        epoch_data = {vec:[] for vec in VECTORS}
        epoch_data['success'] = []
        rewMat2 = []

    if args.render == 0 and args.save_test_result == 1:
        temp_rows = []
        w = open(csvPath, 'at', newline='', encoding='cp949')
        existFile, existCol, start, isAlreadyFin = check_csv_file_exist(csvPath, baseColumns, end=epoch_list[-1])
        if isAlreadyFin:
            print(f'※※※ Already tested : {args.exp_name}')
            w.close()
            return
        
        wr = csv.writer(w)
        if existCol:
            itr_max_epoch = start
            indices = np.where(itr_max_epoch < np.array(epoch_list))
            offset = abs(len(epoch_list)-len(epoch_list[indices]))
            epoch_list = epoch_list[indices]
            xEnvlist2 = xEnvlist2[indices]
        else:
            wr.writerow(baseColumns)
    
    for i, epoch in enumerate(epoch_list):
        if args.render == 0 and args.save_test_result == 1:
            if w.closed:
                w = open(csvPath, 'at', newline='', encoding='cp949')
                wr = csv.writer(w)

        cnt += 1
        if epoch % args.print_freq == 0:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            time_past = (time.time()-time_start) / 3600
            print(f'{current_time}| {time_past:.2f} | {args.exp_name} | epoch = {epoch} | itr = {i+offset}')

        #* load and policy setting 
        policy = load_policy(model_path, epoch)
        set_policy_attributes(policy, args)
        
        episode_step_data, epi_success, epi_rewards, epi_optRew\
        = eval_model(env, policy, 
                    itr=epoch,
                    n_eval_episodes=args.n_eval_episodes,
                    max_env_steps=args.max_env_steps,
                    eval_greedy=args.eval_greedy,
                    render=args.render,
                    inspect_steps=args.inspect_steps,
                    seed=args.seed,
                    flag=flag,
                    )
        
        if flag[0]: # ctrl + c : interrupt
            return
        
        if args.render == 0 and args.save_test_result == 1:
            #* Store step-by-step sequence data separately for each vector in each episode
            temp = {vec:[torch.zeros(args.max_env_steps)] for vec in VECTORS}
            temp_success = [torch.zeros(args.max_env_steps, dtype=torch.int64)]
            for step_success, one_step in episode_step_data:
                temp_success.append(torch.tensor(step_success))
                for vec in VECTORS:
                    temp[vec].append(torch.tensor(one_step[vec]))
                    
            #* Apply padding since step lengths vary across episodes, and store data for each epoch
            temp_success = pad_sequence(temp_success, padding_value=0, batch_first=True).numpy()
            epoch_data['success'].append(temp_success[1:])
            for vec in VECTORS:
                sequences = temp[vec]
                if len(sequences[0].shape) != 0: # if data is not a scalar value
                    sequences = pad_sequence(sequences, padding_value=0, batch_first=True)

                sequences = sequences[1:]
                epoch_data[vec].append(sequences.numpy())
                
            #* Compute the average score across all episodes and save it to an Excel file.
            rewMat2.append(epi_rewards['reward'])

            row = [epoch] + [xEnvlist2[i]] + [np.mean(epi_success)] + [np.mean(epi_rewards[vec]) for vec in VECTORS] + [current_time]

            for c in MATLAB_COLS:
                mat_file[c].append(np.mean(epi_rewards[MATLAB_COLS_MAPPING[c]]))

            mat_file['optRew'].append(np.mean(epi_optRew))
            mat_file['avgSucc'].append(np.mean(epi_success))
            
            temp_rows.append(row)
            if i == 0 or args.save_freq <= cnt:
                cnt = 0
                save_matlab(file_path=mat_path, backup_path=pkl_path,
                            epoch_data=epoch_data, rewMat2=rewMat2, mat_file=mat_file,
                            trAvgReward=trAvgReward, trAvgSuccess=trAvgSuccess, trainMetric=trainMetric,
                            xlist=args.epoch_list, xEnvlist=args.xEnvlist,
                            params=args.func_parameter_parsing,
                            vectors=VECTORS,
                            formats='mat')
                
                for row in temp_rows:
                    wr.writerow(row)
                temp_rows = []
                w.close()

    if args.render == 0 and args.save_test_result == 1:
        if not w.closed: w.close()

        if not isAlreadyFin:
            time.sleep(0.5)
            add_train_avg_result(csvPath, model_path, trAvgReward, colName='train_avg_reward')
            add_train_avg_result(csvPath, model_path, trAvgSuccess, colName='train_avg_success')
            os.remove(f'{pkl_path}.pkl')
    
    if args.device != 'cpu':
        torch.cuda.empty_cache()

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_past = (time.time()-time_start) / 3600
    print(f'{current_time} | {time_past:.2f} | Test finished')


