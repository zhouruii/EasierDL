from torch import nn

from uchiha.models.builder import build_module, MODEL


build = {
    'embedding': build_module,
    'basemodule': build_module,
    'modules': build_module,
    'downsample': build_module,
    'downsamples': build_module,
    'upsample': build_module,
    'upsamples': build_module,
    'head': build_module,
}


def _build_stacks(cfgs):
    """ Build modules that are stacked continuously

    Args:
        cfgs (List[dict]): The list containing config information for building stacks.

    Returns:
        nn.ModuleList: Built stacked modules
    """
    if cfgs is None:
        return
    modules = nn.ModuleList()
    for idx, cfg in enumerate(cfgs):
        module_name, module_cfg = cfg.popitem()
        build_func = build.get(module_name)
        modules.append(build_func(module_cfg))
    return modules


@MODEL.register_module()
class Stack(nn.Module):
    """ Stacked networks

    Args:
        preprocessor (dict): Config information for building the preprocessor. Default: None.
        stacks (List[dict]): The list containing config information for building stacks. Default: None
        postprocessor (dict): Config information for building the postprocessor. Default: None.
    """

    def __init__(self,
                 preprocessor=None,
                 stacks=None,
                 postprocessor=None):

        super().__init__()
        self.preprocessor: nn.Module = build_module(preprocessor)

        self.stacks: nn.ModuleList = _build_stacks(stacks)

        self.postprocessor: nn.Module = build_module(postprocessor)

    def forward(self, x):
        # preprocess
        out = self.preprocessor(x) if self.preprocessor else x

        # stacks
        for module in self.stacks:
            out = module(out)

        # postprocessor
        out = self.postprocessor(out) if self.postprocessor else out

        return out
