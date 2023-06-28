import yaml 
from npbg.utils.arguments import eval_modules, eval_functions
# import hiyapyco

def save_yaml(what, where):
    with open(where, 'w') as f:
        f.write(yaml.dump(what, default_flow_style = False))
        
def load_yaml(where, eval_data=False):
    with open(where, 'r') as f:
        data=yaml.load(f)
    if eval_data:
        eval_modules(data)
        eval_functions(data)
    return data