# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import os
import subprocess
import typing as tp
from dataclasses import dataclass
from random import randint
from re import M

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from examples.nllb.nllb_lib.nllb_module import DistributedRequirements, NLLBModule


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


@dataclass
class MainConfig:
    cfg: TrainConfig = TrainConfig()


class TrainModule(NLLBModule):
    def __init__(self, config):
        super().__init__(config)

        # values in cfg configurable through entire .yaml files in conf/cfg/
        cfg = config.cfg

        self.output_dir = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print("TRAINING DIR: ", self.output_dir)

        config_yaml_file = os.path.join(self.output_dir, "train_script.yaml")
        with open(config_yaml_file, "w") as f:
            f.write(OmegaConf.to_yaml(config.cfg))

        cluster_name = cfg.cluster.cluster_name
        assert cluster_name in cfg.dataset.data_prefix
        data_prefix = cfg.dataset.data_prefix[cluster_name]
        assert data_prefix is not None
        assert os.path.isdir(data_prefix), f"{data_prefix} is not a directory"
        assert os.access(data_prefix, os.R_OK), f"{data_prefix} is not readable"

        data_dir = ""
        for shard_id in range(cfg.dataset.num_shards):
            data_dir += f":{data_prefix}/shard{shard_id:03d}"
        data_dir = data_dir[1:]  # remove leading colon
        print("data_dir: ", data_dir)
        self.data_dir = data_dir

    def launch_job(self):
        # values in cfg configurable through entire .yaml files in conf/cfg/
        cfg = self.config.cfg

        if cfg.dataset.langs is None:
            assert cfg.dataset.langs_file is not None
            langs = os.path.join(cfg.fairseq_root, cfg.dataset.langs_file)
        else:
            langs = cfg.dataset.langs

        if cfg.dataset.lang_pairs is None:
            assert cfg.dataset.lang_pairs_file is not None
            lang_pairs = os.path.join(cfg.fairseq_root, cfg.dataset.lang_pairs_file)
        else:
            lang_pairs = cfg.dataset.lang_pairs

        tensorboard_dir = os.path.join(self.output_dir, "tb")

        checkpoint_activations_param = (
            "--checkpoint-activations" if cfg.checkpoint_activations else ""
        )
        zero2_param = "--zero2" if cfg.zero2 else ""

        sweep_command = f"""
            cd {cfg.fairseq_root}
            python -m {cfg.module_name} \
                -d {self.data_dir} \
                -p {cfg.train_prefix} \
                --checkpoints-dir {self.output_dir} \
                --partition {cfg.cluster.partition} \
                -t {cfg.num_trials} \
                -n {cfg.num_nodes} \
                -g {cfg.num_gpus_per_node} \
                --resume-failed \
                --time {cfg.max_time_mins} \
                --mem {cfg.mem} \
                --sampling-method temperature \
                --sampling-temperature {cfg.temp} \
                --decoder-langtok \
                --encoder-langtok {cfg.encoder_langtok} \
                --langs {langs} \
                --lang-pairs {lang_pairs} \
                --moe-eval-cap {cfg.moe_eval_cap} \
                --ddp-backend {cfg.ddp_backend} \
                --max-update {cfg.max_updates} \
                --max-tokens {cfg.max_tokens} \
                --update-freq {cfg.update_freq} \
                --warmup-updates {cfg.warmup} \
                --lr {cfg.lr} \
                --opt adam16bit \
                --share-all-embeddings \
                --save-interval-updates 5000 \
                --tensorboard-logdir {tensorboard_dir} \
                --arch {cfg.arch} \
                --dropout {cfg.dropout} \
                --validate-interval-updates {cfg.validate_interval_updates} \
                --seed {cfg.seed} \
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


@hydra.main(config_path="conf", config_name="base_config")
def main(config: DictConfig) -> None:
    train_module = TrainModule(config)
    train_module.launch_job()


if __name__ == "__main__":
    main()
