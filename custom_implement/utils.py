import re
from send2trash import send2trash

def get_mapsize(exp_name):
    maps = re.compile('map[0-9]+').search(exp_name).group()
    return maps

def num(s):
    if s is None:
        return None
    
    n = re.compile('-?[0-9.]+').search(s)
    if n != None:
        n = n.group()
        if int(n) == float(n):
                n = int(n)
        else:
            n = float(n)
    return n

def extracting_number(string, type_=None, n_elem=0):
    nums = re.compile('-?[0-9.]+')
    numbers = nums.findall(string)
    numbers = list(map(type_, numbers))
    for i, n in enumerate(numbers):
        if float(n) == int(n):
            numbers[i] = int(n)
            
    return numbers

def pill_last_path(path):
    return path.split('\\')[-1]


def send_to_trash(file_path=None):
    # Move file to the trash using send2trash.
    try:
        send2trash(file_path)
        print(f"{file_path} has been moved to the trash.")
    except FileNotFoundError:
        print(f"{file_path} does not exist.")
    except Exception as e:
        print(f"Error occurred: {e}")

def get_model_on_what_device(*args, **kwargs):
    # Print args.
    for arg in args:
        print(f"{arg}: {str(arg)}")

    # Print kwargs.
    prt = False
    for key, value in kwargs.items():
        if key == 'device':
            device = value
        if key == 'var':
            var = value
        if key == 'print_option':
            print_option = value
            if value == True:
                prt = True

    if prt:
        print_option = True
    else:
        print_option = False

    for key, model in kwargs.items():
        #print(f"{key}: {str(model)}")
        if key == 'model':
            if hasattr(model, '_parameters'):
                p = next(model.parameters())
                device_on = f'{p.device.type}:{p.device.index}' if p.device.index != None else 'cpu'

                p = None
                class_ = str(model.__class__).replace('>', '').split('.')[-1].replace('\'', '')
                if print_option:
                    print(f'{var}: {class_} is on \"{device_on}\"', end=' ')
                    if device_on == device:
                        print('...... OK')
                    else:
                        print('...... Fail')
            else:
                print(f'{var}: {class_} is not Neural Network')

            break
        
def set_policy_attributes(policy, args):
    policy._n_agents = args.n_agents
    policy.p_loss = args.tepl

    # set devices
    device = args.device
    set_polcy_device_with(policy, device, print_option=False)

    if args.torch_tensor_type == 'float':
        policy.float()
    else:
        policy.double()
        
        
def set_polcy_device_with(policy, device, print_option=False):
    policy.to(device)
    policy.device = device
    get_model_on_what_device(var='policy', model=policy, device=device, )

def set_nn_device_with(algo, device, print_option=True):
    #! set device to Networks
    print(f'######################################################################################################')
    #print(f'######################################################################################################')
    print(f'Training Device Setting With: {device}')
    algo.device = device

    algo._optimizer.device = device
    algo._baseline_optimizer.device = device

    algo.baseline.to(device)
    algo.baseline.device = device
    if hasattr(algo.baseline, 'module'):
        algo.baseline.module.device = device
        algo.baseline.module.to(device)
        get_model_on_what_device(var='algo.baseline.module', model=algo.baseline.module, device=device, print_option=print_option)

    get_model_on_what_device(var='algo.baseline', model=algo.baseline, device=device, print_option=print_option)
    algo.policy.to(device)
    algo.policy.device = device
    get_model_on_what_device(var='algo.policy', model=algo.policy, device=device, )
    

    algo._old_policy.to(device)
    algo._old_policy.device = device
    get_model_on_what_device(var='algo._old_policy', model=algo._old_policy, device=device, print_option=print_option)

    #print(f'######################################################################################################')
    print(f'######################################################################################################')
    #! set device to Networks----------------------------------------------------------------------

def check_nn_on_device(algo, device, print_option=True):
    #! check Networks on device
    #print(f'######################################################################################################')
    print(f'######################################################################################################')
    print(f'Check Device Setting With: {device}')
    get_model_on_what_device(var='algo.baseline', model=algo.baseline, device=device, print_option=print_option)
    get_model_on_what_device(var='algo.policy', model=algo.policy, device=device, print_option=print_option)
    get_model_on_what_device(var='algo._old_policy', model=algo._old_policy, device=device, print_option=print_option)
    #print(f'######################################################################################################')
    print(f'######################################################################################################')
    #! check Networks on device ----------------------------------------------------------------------
    
