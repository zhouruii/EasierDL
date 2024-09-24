from torch import nn

from ..builder import (build_preprocessor, build_embedding, build_basemodule,
                       build_downsample, build_head, MODEL, build_model, build_postprocessor)
from ...utils.misc import strings_to_list


# TODO 待完善
@MODEL.register_module()
class ParallelSpatialChannelTransformer(nn.Module):

    def __init__(self,
                 embedding=None,
                 parallels=None,
                 postprocessor=None):
        super().__init__()
        self.embedding = build_embedding(embedding)

        self.workflows = nn.ModuleList()
        for workflow in parallels:
            self.workflows.append(build_model(workflow))

        self.postprocessor = build_postprocessor(postprocessor)

    def forward(self, x):

        x = self.embedding(x)
        x_parallel = [x, x]

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
