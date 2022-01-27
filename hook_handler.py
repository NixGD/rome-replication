from typing import Any, Callable, Hashable

import torch as t
from torch import nn
from torch.utils.hooks import RemovableHandle


class HookHandler:
    def __init__(self):
        self.reset_data()

    def reset(self):
        for h in self.hook_handles:
            h.remove()
        self.reset_data()

    def reset_data(self):
        self.activations = {}
        self.inputs = []
        self.outputs = []
        self.hook_handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset()
        # print("All hooks removed!")

    def add_hook(
        self,
        mod: nn.Module,
        fn: Callable[[nn.Module, Any, t.Tensor], Any],
    ):
        self.hook_handles.append(mod.register_forward_hook(fn))

    def add_save_activation_hook(
        self,
        mod: nn.Module,
        key: Hashable,
    ):
        def fn(model, input, output):
            self.activations[key] = output.detach()

        self.hook_handles.append(mod.register_forward_hook(fn))

    def add_save_input_hook(
        self,
        mod: nn.Module
    ):
        def fn(model, input, output):
            self.inputs.append(input[0].detach())
        
        self.hook_handles.append(mod.register_forward_hook(fn))
    
    def add_save_input_output_hook(
        self,
        mod: nn.Module
    ):
        def fn(model, input, output):
            self.inputs.append(input[0].detach())
            self.outputs.append(output.detach())

        self.hook_handles.append(mod.register_forward_hook(fn))

class SaveAllActivations(HookHandler):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._activations = [None]*len(model.blocks)

    def get_add_act_hook(self, i):
        def add_act(model, input, output):
            self._activations[i] = output[0]
        return add_act

    def __enter__(self):
        for i, block in enumerate(self.model.blocks):
            self.add_hook(block, self.get_add_act_hook(i))
        return self

    def get_activations(self):
        """Returns tensor of shape [seq_len, layers, hidden_size]"""
        assert not any([(a is None) for a in self._activations]), self._activations
        return t.stack(self._activations, dim=1)