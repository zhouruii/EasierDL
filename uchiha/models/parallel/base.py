from torch import nn

from ..builder import (build_preprocessor, build_embedding, build_basemodule,
                       build_downsample, build_head, MODEL, build_model, build_postprocessor)
from ...utils.misc import strings_to_list


@MODEL.register_module()
class Parallel(nn.Module):
    """ parallel pipeline work

    It is divided into three parts, first preprocessing the data,
    then sending it to multiple pipelines (each pipeline is a `Stack` operation),
    and finally using post-processing to integrate the output of multiple pipelines.

    Args:
        preprocessor (dict): Config information for building the preprocessor
        parallels (List[dict]): A list of config information for building the parallel,
            each config information represents a `Stack`
        postprocessor (dict): Config information for building the postprocessor
    """

    def __init__(self,
                 preprocessor=None,
                 parallels=None,
                 postprocessor=None):

        super().__init__()
        self.preprocessor: nn.Module = build_preprocessor(preprocessor)

        self.workflows = nn.ModuleList()
        for workflow in parallels:
            self.workflows.append(build_model(workflow))

        self.postprocessor: nn.Module = build_postprocessor(postprocessor)

    def forward(self, x):
        if self.preprocessor:
            x_parallel = self.preprocessor(x)
        else:
            x_parallel = x

        # core
        y_parallel = []
        for idx, x in enumerate(x_parallel):
            y = self.workflows[idx](x)
            y_parallel.append(y)

        if self.postprocessor:
            out = self.postprocessor(y_parallel)
        else:
            out = y_parallel

        return out

