# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
from re import M
import subprocess
import typing as tp
from dataclasses import dataclass
from random import randint

import hydra
from omegaconf import MISSING, DictConfig

from examples.nllb.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)


@dataclass
class ClusterConfig:
    cluster_name: str = MISSING
    data_dir: str = MISSING
    partition: str = MISSING
    memory_multiplier: int = 0
    timeout_min: int = 1000


@dataclass
class DatasetConfig:
    dataset_name: str = MISSING
    num_shards: int = MISSING
    langs: str = MISSING
    langs_file: str = MISSING
    lang_pairs: str = MISSING
    lang_pairs_file: str = MISSING
    data_prefix: tp.Dict[str, str] = MISSING


@dataclass
class ModelTypeConfig:
    name: str = MISSING
    moe_params: str = MISSING
    expert_count: int = MISSING


@dataclass
class TrainConfig:
    cluster: ClusterConfig = ClusterConfig()
    dataset: DatasetConfig = DatasetConfig()
    model_type: ModelTypeConfig = ModelTypeConfig()
    fairseq_root: str = MISSING
    output_dir: str = MISSING
    train_prefix: str = MISSING
    seed: int = MISSING
    arch: str = MISSING
    max_updates: int = MISSING
    validate_interval_updates: int = MISSING
    encoder_langtok: str = MISSING
    ddp_backend: str = MISSING
    fp16: bool = MISSING
    lr: float = MISSING
    warmup: int = MISSING
    max_tokens: int = MISSING
    update_freq: int = MISSING
    num_nodes: int = MISSING
    num_gpus_per_node: int = MISSING
    temp: float = MISSING
    dropout: float = MISSING
    module_name: str = "examples.nllb.modeling.sweep.sweep_mmt"
    num_trials: int = 1
    max_time_mins: int = 4320
    mem: int = 0
    moe_eval_cap: float = 1.0
    checkpoint_activations: bool = False
    zero2: bool = False


class TrainModule(NLLBModule):
    def __init__(self, config):
        super().__init__(config)
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print("TRAINING DIR: ", self.output_dir)

        cluster_name = config.cluster.cluster_name
        assert cluster_name in config.dataset.data_prefix
        data_prefix = config.dataset.data_prefix[cluster_name]
        assert data_prefix is not None
        assert os.path.isdir(data_prefix), f"{data_prefix} is not a directory"
        assert os.access(data_prefix, os.R_OK), f"{data_prefix} is not readable"

        data_dir = ""
        for shard_id in range(config.dataset.num_shards):
            data_dir += f":{data_prefix}/shard{shard_id:03d}"
        data_dir = data_dir[1:]  # remove leading colon
        print("data_dir: ", data_dir)
        self.data_dir = data_dir

    def launch_job(self):
        config = self.config

        if config.dataset.langs is None:
            assert config.dataset.langs_file is not None
            langs = os.path.join(config.fairseq_root, config.dataset.langs_file)
        else:
            langs = config.dataset.langs

        if config.dataset.lang_pairs is None:
            assert config.dataset.lang_pairs_file is not None
            lang_pairs = os.path.join(
                config.fairseq_root, config.dataset.lang_pairs_file
            )
        else:
            lang_pairs = config.dataset.lang_pairs

        tensorboard_dir = os.path.join(self.output_dir, "tb")

        checkpoint_activations_param = (
            "--checkpoint-activations" if config.checkpoint_activations else ""
        )
        zero2_param = "--zero2" if config.zero2 else ""

        sweep_command = f"""
            cd {config.fairseq_root}
            python -m {config.module_name} \
                -d {self.data_dir} \
                -p {config.train_prefix} \
                --checkpoints-dir {self.output_dir} \
                --partition {config.cluster.partition} \
                -t {config.num_trials} \
                -n {config.num_nodes} \
                -g {config.num_gpus_per_node} \
                --resume-failed \
                --time {config.max_time_mins} \
                --mem {config.mem} \
                --sampling-method temperature \
                --sampling-temperature {config.temp} \
                --decoder-langtok \
                --encoder-langtok {config.encoder_langtok} \
                --langs {langs} \
                --lang-pairs {lang_pairs} \
                --moe-eval-cap {config.moe_eval_cap} \
                --ddp-backend {config.ddp_backend} \
                --max-update {config.max_updates} \
                --max-tokens {config.max_tokens} \
                --update-freq {config.update_freq} \
                --warmup-updates {config.warmup} \
                --lr {config.lr} \
                --opt adam16bit \
                --share-all-embeddings \
                --save-interval-updates 5000 \
                --tensorboard-logdir {tensorboard_dir} \
                --arch {config.arch} \
                --dropout {config.dropout} \
                --validate-interval-updates {config.validate_interval_updates} \
                --seed {config.seed} \
                --snapshot-code \
                --use-local-shard-size \
                {checkpoint_activations_param} \
                {zero2_param}
        """

        print("RUNNING SWEEP COMMAND:")
        print(sweep_command)

        subprocess.run(
            sweep_command, shell=True, check=True,
        )

    async def run(
        self, iteration_value: tp.Optional[tp.Any] = None, iteration_index: int = 0,
    ):
        # launching one training job synchronously for now
        pass


@hydra.main(config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    train_module = TrainModule(config)
    train_module.launch_job()


if __name__ == "__main__":
    main()
