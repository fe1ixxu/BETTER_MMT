# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from examples.nllb import nllb_lib

logger = logging.getLogger("nllb_module")

################################################################################
#  Module definition
################################################################################


class Requirements(ABC):
    pass


@dataclass
class DistributedRequirements(Requirements):
    nodes: int = 1
    mem_gb: int = 200
    tasks_per_node: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 5
    timeout_min: int = 720
    constraint: tp.Optional[str] = None


class LocalOnlyRequirements(Requirements):
    pass


class NLLBModule(ABC):
    @staticmethod
    def build(config: tp.Any, **kwargs):
        """
        given a loaded config with a _target_ and a config entry, build the
        correct module.
        """
        merged_conf = OmegaConf.merge(config, {"config": kwargs})

        # hydra is good at that.
        return hydra.utils.instantiate(
            merged_conf,
            _recursive_=False,
        )

    def __init__(self, config: tp.Any):
        self.config = config
        if not isinstance(config, DictConfig):
            self.config = OmegaConf.structured(config)
        OmegaConf.resolve(self.config)
        OmegaConf.set_readonly(self.config, True)

    def __call__(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        """
        called when the job starts running.
        please implement `run` instead as we might need to add generic stuff in here
        """
        res = self.run(iteration_value=iteration_value, iteration_index=iteration_index)
        if not isinstance(res, tp.Coroutine):
            # Return result value in case of synchronous method call
            return res

        # Handle async `run` implementation
        have_event_loop = True
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 'RuntimeError: There is no current event loop.
            have_event_loop = False

        # TODO: Explain more when we can return a coroutine
        # This is weird: depending on the context we either return a result
        # or a coroutine.
        if have_event_loop:
            # this should be awaited by whoever is calling the raw module
            return res
        else:
            # We are in a separate process, run it with asyncio
            return asyncio.run(res)

    @abstractmethod
    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> tp.Any:
        """
        the core of your module, implement your logic here.

        If `array` returned an array, this is an array job and this `run` be
        called for each iteration, the value of the specific iteration in the array
        will be passed down to you. If it's not an array job, this will be None.
        """
        ...

    def array(self) -> tp.Optional[tp.List[tp.Any]]:
        """
        if you want to submit your job as an array job, you can compute
        the array of values you want to process here. This will be processed
        before the job is submitted to the particular cluster chosen.

        By default, we return None to indicate this is not an array job.
        """
        return None

    def requirements(self) -> Requirements:
        """
        return a set of Requirements for your module, like num of gpus etc.
        If you return None, this will be launched "inline" without scheduling any new job.
        """
        return LocalOnlyRequirements()

    def name(self):
        """
        implement this if you want to give a fancy name to your job
        """
        # TODO ideally use hydra override_dirname here
        return "_".join([self.__class__.__name__, self.sha_key()])

    def cache_key(self):
        return (
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.version(),
            self.config,
        )

    # TODO: @functools.cached_property()
    # This is only safe if cache_key is not allowed to change, in particular if config is frozen.
    # Can we guarantee that ?
    def sha_key(self):
        return nllb_lib.utils.sha_key(repr(self.cache_key()))

    def comment(self):
        """
        same as `name` but for the job comment
        """
        return None

    @classmethod
    def version(cls):
        """
        the version of the module. If you want to invalidate
        some cache, you can change that
        """
        return "0.0"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        """
        Validate the output of this module (for a single step of the array if it's an array module).
        This is called to invalidate the cache when needed but also at the end of the module run for
        sanity check.
        You can either return False or Raise/throw an exception if the results are not valid.

        The default implementation checks if the content of pickle is a "Path",
        and invalidate cache if the corresponding file is gone.
        """
        if isinstance(output, Path) and not output.exists():
            logger.warning(
                f"Cache for: {self.name()} iteration {iteration_index}"
                f"points to missing file {output}, will invalidate it."
            )
            return False
        return True