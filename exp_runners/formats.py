import argparse
import re
import collections
import os
from glob import glob
import pandas as pd
from custom_implement.utils import extracting_number, pill_last_path
from custom_implement.utils import send_to_trash
from testing import TEST_METRIC_VECTORS

class FileFormat():
    rTrainForm = re.compile(r'(cent|comm|obs)*.*TR')
    rTestQuery = re.compile(r'(cent|comm|obs)*.*TE')
    def __init__(self):
        pass
    
    @classmethod
    def getTrainForm(self, exp_name):
        """
        Extract only up to the train setup part from the file name.
        """

        trainForm = self.rTrainForm.search(exp_name).group()
        
        if trainForm:
            while trainForm[-1] in ['_', '.']:
                trainForm = trainForm[:-1]

            args = argparse.Namespace()
            args.exp_name = exp_name

            args = self.parameter_parsing(exp_name)
            args.trainForm = trainForm
            
            return trainForm
        
    @classmethod
    def getTestForm(self, query):
        testForm = self.rTestQuery.search(query).group()
        if testForm:
            while testForm[-1] in ['_', '.']:
                testForm = testForm[:-1]
            return testForm
        else:
            pass

    @classmethod
    def get_trained_list(self, threshold=0):
        trained = []
        if os.path.exists(self.trPath) and os.path.isdir(self.trPath):
            pass
        else:
            raise Exception(f"The trained model path at '{self.trPath}' does not exist.")

        train_dirs = glob(f'{self.trPath}/*')
        if len(train_dirs) == 0:
            return trained
        
        for model_name in train_dirs:
            if threshold != 'debug':
                thrEpoch = threshold
            else:
                thrEpoch = 1

            itrs = glob(f'{model_name}/itrs/itr_*.pkl')
            
            if thrEpoch<=len(itrs):
                itrs_sorted = sorted(itrs) # Sort file names in numerical order.
                itr_max = re.compile('itr_[0-9]+').findall(itrs_sorted[-1])[0]
                itr_max = int(re.compile('[0-9]+').findall(itr_max)[0])

                if thrEpoch <= itr_max :
                    # really trained and finished well
                    trainForm = self.getTrainForm(pill_last_path(model_name))

                    if trainForm not in trained:
                        trained.append(trainForm)
                else:

                    print(model_name)
                    raise Exception('error occur during training')
            
            else:
                pass
            
        return trained
    
    @classmethod
    def get_tested_list(self, args, n_threshold=50):
        tested = []
        n_threshold = 1 if n_threshold == 'debug' else n_threshold
        
        if not(os.path.exists(self.tePath) and os.path.isdir(self.tePath)):
            raise Exception(f"The trained model path at '{self.trPath}' does not exist.")

        test_files = glob(f'{self.tePath}/*.csv')
        for teFile in test_files:
            try:
                df_test = pd.read_csv(f'{teFile}')
            except pd.errors.EmptyDataError:
                continue
            
            valid = True
            columns = df_test.columns.tolist()
            for vector in TEST_METRIC_VECTORS:
                if vector in columns: continue
                else:
                    valid = False
                    break

            if valid:
                if n_threshold <= len(df_test):
                    testForm = self.getTestForm(teFile)
                    tested.append(testForm)

        return tested
    
    @classmethod
    def parameter_parsing(self, args):
        if type(args) == argparse.Namespace:
            exp_name = args.exp_name
        elif type(args) == str:
            exp_name = args
            args = argparse.Namespace()

        args_dict=vars(args)

        rExe = re.compile(r'(cent|comm|obs)')
        rScen = re.compile(r'(pp|co)')
        rNGroup= re.compile('Ng[0-9]+')
        rResidual_conn = re.compile('resi-?[0-9]+')
        # rEncoder = re.compile('enc[0-9]+')
        rSeed = re.compile('seed[0-9]+')
        
        rRewards = re.compile('rew[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        rDensity = re.compile('den[0-9]+_[0-9]+_[0-9]+')
        
        rMap = re.compile('map[0-9]+')
        rNa = re.compile('na[0-9.]+')
        rNt = re.compile('nt[0-9.]+')
        rLoad = re.compile('load[0-9.]+')
        rHop = re.compile('L[0-9]+')
        rRsen = re.compile('Rsen[0-9]+')
        
        rTrRcom = re.compile('trRcom[0-9]+')
        rTrPloss = re.compile('trpl[0-9.]+')
        rTrPfault = re.compile('trpf[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        rTmax = re.compile('Tmax[0-9]+')
        
        rTeRcom = re.compile('teRcom[0-9]+')
        rTePloss = re.compile('tepl[0-9.]+')
        rTePfault = re.compile('tepf[0-9.]+_[0-9.]+_[0-9.]+_[0-9.]+')
        
        keys = {
                'exe': (rExe, str, 1),
                'scenario': (rScen, str, 1),
                'n_groups': (rNGroup, int, 1),
                
                'residual_conn': (rResidual_conn, int, 1),
                'seed': (rSeed, int, 1),
            
                'rewards': (rRewards, float, 4),
                'density': (rDensity, int, 3),
                
                'grid_size':(rMap, int, 1),
                'n_agents':(rNa, float, 1),
                'n_preys':(rNt, float, 1),
                'load':(rLoad, float, 1),
                'n_gcn_layers':(rHop, int, 1),
                'Rsen':(rRsen, int, 1),

                'trRcom': (rTrRcom, float, 1),
                'trpl': (rTrPloss, float, 1),
                'trpf': (rTrPfault, float, 4),
                'max_env_steps': (rTmax, int, 1),
                
                'teRcom': (rTeRcom, float, 1),
                'tepl': (rTePloss, float, 1),
                'tepf': (rTePfault, float, 4), 
                }

        for k, (rex, dtype_, n_elem) in keys.items():
            rst = rex.search(exp_name)
            if rst:
                rst = rst.group()
            else:
                continue
            
            if rst:
                if dtype_==str:
                    args_dict[k] = dtype_(rst)
                    continue
                
                number = extracting_number(rst,dtype_,n_elem)
                
                if k == 'index':
                    args_dict[k] = f'ID{str(number[0]).zfill(3)}'
                    continue
                
                if len(number) == n_elem == 1:
                    args_dict[k] = number[0]
                
                else:
                    args_dict[k] = number
                    if k == 'rewards':
                        args_dict['capture_reward'] = dtype_(number[0])
                        args_dict['step_cost'] = dtype_(number[1])
                        args_dict['rm'] = dtype_(number[2])
                        args_dict['penalty'] = dtype_(number[3])

                    if k in ['trpf', 'tepf']:
                        PfNmin, PfNmax, PfPmin, PfPmax= dtype_(number[0]), dtype_(number[1]), dtype_(number[2]), dtype_(number[3])
                        args_dict[k+'Nmin'] = PfNmin 
                        args_dict[k+'Nmax'] = PfNmax 
                        args_dict[k+'Pmin'] = PfPmin 
                        args_dict[k+'Pmax'] = PfPmax
                    
                
        d = vars(args)
        for k, v in d.items():
            try:
                if type(v) != str:
                    if float(v) == int(v):
                        d[k] = int(v)
            except:
                continue
                
        return args

    @classmethod
    def process_exp_name_default(self, args):
        exp_layout = collections.OrderedDict([
            ('{}', f'{args.exe}_{args.scenario}'),
            ('Ng{}', args.n_groups),
            ('resi{}', args.residual),
            ('seed{}', args.seed),
            
            ('rew{}', f'{args.capture_reward}_{args.step_cost}_{args.rm}_{args.penalty}'),
            ('den{}', f'{args.density[0]}_{args.density[1]}_{args.density[2]}'),
            
            ('map{}', args.grid_size),
            ('na{}', args.n_agents),
            ('nt{}', args.n_preys),
            ('load{}', args.load),
            
            ('L{}', args.n_gcn_layers),
            ('Rsen{}', args.Rsen),
            
            ('trRcom{}', args.trRcom),
            ('trpl{}', args.trpl),
            ('trpf{}', f'{args.trpfNmax}_{args.trpfNmin}_{args.trpfPmax}_{args.trpfPmin}'),
            ('Tmax{}', args.max_env_steps),
            ('TR', ''),
            
            ('teRcom{}', args.teRcom),
            ('tepl{}', args.tepl),
            ('tepf{}', f'{args.tepfNmax}_{args.tepfNmin}_{args.tepfPmax}_{args.tepfPmin}'),
            ('TE', ''),
        ])
        exp_name = '_'.join([key.format(val) for key, val in exp_layout.items()])
        return exp_name

    def preSetting(self, args, scenario_name):
        if args.exp_name is None:
            exp_name = self.process_exp_name_default(args)
        else:
            exp_name = args.exp_name

        args.prefix = scenario_name
        id_suffix = ('_' + str(args.run_id)) if args.run_id != 0 else ''
        unseeded_exp_dir = './data/' + args.loc +'/' + exp_name[:-7]
        args.exp_dir = './data/' + args.loc +'/' + exp_name + id_suffix

        if args.mode == 'restore':
            if os.path.isfile(args.exp_dir + '/params.pkl'):
                pass
            else:
                send_to_trash(args.exp_dir)
                args.mode = 'train'

        # Enforce
        args.center_adv = False if args.entropy_method == 'max' else args.center_adv

        return args