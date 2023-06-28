import torch
import torch.nn as nn
    
            
class DynamicModule(nn.Module):
    def __init__(self, submodules, ordered_keys):
        super().__init__()
        '''
        submodules: dict
        '''
        assert isinstance(submodules, dict)

        self.id_module_map = {}
        
        self.ordered_keys = ordered_keys

        self.submodules = {k: v.cpu() for k, v in submodules.items()}

    def load_modules(self, submodule_ids):
        if torch.is_tensor(submodule_ids):
            submodule_ids = submodule_ids.cpu().tolist()
        elif isinstance(submodule_ids, int):
            submodule_ids = [submodule_ids]

        self.id_module_map = {}
        for sid in submodule_ids:
            name = self.ordered_keys[sid]
            self._modules[name] = self.submodules[name]
            self.id_module_map[sid] = name

    def unload_modules(self):
        for name in self.id_module_map.values():
            del self._modules[name]
            