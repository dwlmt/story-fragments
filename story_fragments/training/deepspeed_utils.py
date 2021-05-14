import json
import logging
import os
import tempfile
from typing import Union, Dict, Any

import torch
from deepspeed.utils import logger as ds_logger

ds_logger.setLevel(logging.WARNING)
ds_logger.propagate = False
import deepspeed

from allennlp.common import FromParams

JsonDict = Dict[str, Any]


class DeepspeedConfig(FromParams):
    def __init__(
            self,
            optimizer: JsonDict,
            fp16: JsonDict = {'enabled': False},
            amp: JsonDict = {'enabled': False},
            zero_optimization: Union[bool, Dict] = False,
            activation_checkpointing: Union[Dict] = {'partition_activations': False},
            scheduler: Union[Dict] = {},
            zero_allow_untested_optimizer: bool = True,
            gradient_clipping: float = 1.0,
            fp32_allreduce: bool = True,
            prescale_gradients: bool = False,
            dump_state: bool = True,
            wall_clock_breakdown: bool = True,
    ):
        self.optimizer = optimizer
        self.fp16 = fp16
        self.amp = amp
        self.zero_optimization = zero_optimization
        self.activation_checkpointing = activation_checkpointing
        self.scheduler = scheduler
        self.zero_allow_untested_optimizer = zero_allow_untested_optimizer
        self.gradient_clipping = gradient_clipping
        self.fp32_allreduce = fp32_allreduce
        self.prescale_gradients = prescale_gradients
        self.dump_state = dump_state
        self.wall_clock_breakdown = wall_clock_breakdown

    @staticmethod
    def build_deepspeed_args(deepspeed_config_path: str, local_rank: int = 0):
        from argparse import ArgumentParser, Namespace
        parser = ArgumentParser()
        parser.add_argument('--local_rank', type=int, default=local_rank)
        parser = deepspeed.add_config_arguments(parser)

        args, _ = parser.parse_known_args()
        arg_dict = vars(args)

        arg_dict.update(dict(deepspeed_config=deepspeed_config_path, deepspeed=True, local_rank=local_rank))
        return Namespace(**arg_dict)

    @property
    def config(self):
        return vars(self)

    def _to_temp_file(self, serialization_dir, **kwargs):
        fd, path = tempfile.mkstemp(dir=serialization_dir)

        config = {**self.config, **kwargs}
        with os.fdopen(fd, 'w') as f:
            f.write(json.dumps(config))

        return path

    def launch(
            self,
            model: torch.nn.Module,
            optimizer: Union[str, torch.optim.Optimizer],
            local_rank: int,
            serialization_dir: str,
            batch_size: int,
            gradient_accumulation_steps: int
    ):
        path = self._to_temp_file(serialization_dir, train_batch_size=batch_size,
                                  gradient_accumulation_steps=gradient_accumulation_steps)

        os.environ["LOCAL_RANK"] = f"{local_rank}"

        ds = deepspeed.initialize(
            args=self.build_deepspeed_args(path, local_rank),
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False
        )

        os.remove(path)
        return ds
