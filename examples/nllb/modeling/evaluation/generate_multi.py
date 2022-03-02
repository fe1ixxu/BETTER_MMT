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

import hydra
from omegaconf import MISSING, DictConfig

from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)


@dataclass
class ClusterConfig:
    name: str = MISSING
    data_dir: str = MISSING
    partition: str = MISSING
    constraint: str = ""
    flores_path: str = MISSING
    memory_multiplier: int = 0
    timeout_min: int = 1000


@dataclass
class LangList:
    bin_root: str = MISSING
    all: tp.List[str] = MISSING
    sampled: tp.List[str] = MISSING


@dataclass
class GenerateMultiConfig:
    model_type: str = "dense"
    direction: str = "en_to_many"
    # use group: fair or azure
    cluster: ClusterConfig = ClusterConfig()
    langs: LangList = LangList()
    # all or sample
    eval_on: str = "sampled"
    model_folder: tp.List[str] = MISSING
    checkpoints: tp.List[str] = MISSING
    gen_splits: tp.List[str] = MISSING
    spm_model: str = MISSING
    data: str = MISSING
    encoder_langtok: str = "tgt"
    output_dir: str = MISSING
    beam_size: int = 4
    fp16: bool = False
    fairseq_root: str = "."


@dataclass
class JobConfig:
    gen_split: str
    checkpoint: str
    lang: str


class GenerateMultiModule(NLLBModule):
    def __init__(self, config):
        super().__init__(config)
        assert os.path.isdir(
            config.model_folder
        ), f"{config.model_folder} is not a directory"
        assert os.access(
            config.model_folder, os.R_OK
        ), f"{config.model_folder} is not a readable"

    def array(self):
        langs = (
            self.config.langs.all
            if self.config.eval_on == "all"
            else self.config.langs.sampled
        )
        return [
            JobConfig(gen_split=split, checkpoint=chk, lang=lang)
            for split in self.config.gen_splits
            for chk in self.config.checkpoints
            for lang in langs
        ]

    def requirements(self):
        if self.config.model_type == "moe":
            gpus = 8
            req = DistributedRequirements(
                tasks_per_node=1,
                nodes=1,
                gpus_per_node=gpus,
                cpus_per_task=gpus * 10,
                mem_gb=gpus * self.config.cluster.memory_multiplier,
                timeout_min=self.config.cluster.timeout_min,
                constraint=self.config.cluster.constraint,
            )
            return req
        else:
            return DistributedRequirements(
                tasks_per_node=1,
                nodes=1,
                gpus_per_node=8,
                cpus_per_task=8,
                mem_gb=48,
                timeout_min=self.config.cluster.timeout_min,
                constraint=self.config.cluster.constraint,
            )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        job_config = iteration_value
        if self.config.direction == "en_to_many":
            src = "eng"
            tgt = job_config.lang
        else:
            src = job_config.lang
            tgt = "eng"

        if self.config.model_type == "moe":
            max_sentences = 16
            cap = 1.0 if self.config.direction == "en_to_many" else 0.5
            req = self.requirements()
            world_size = req.nodes * req.gpus_per_node
            port = (randint(0, 32767) % 119) + 15_000
            model_overrides = {
                "world_size": world_size,
                "moe_eval_capacity_token_fraction": cap,
                "use_moe_pad_mask": False,
            }
            moe_params = (
                "--is-moe"
                f"--distributed-world-size {world_size}"
                f"--distributed-port {port}"
                f"--model-overrides {repr(model_overrides)}"
            )
        else:
            moe_params = ""
            max_sentences = 50
            cap = "no_cap"

        out_dir = os.path.join(
            self.config.output_dir,
            f"gen_output{cap}",
            f"{src}-{tgt}_{job_config.checkpoint}_{job_config.gen_split}",
        )
        os.makedirs(out_dir, exist_ok=True)

        model = os.path.join(self.config.model_folder, f"{job_config.checkpoint}.pt")
        # TODO maybe call generate main directly here with a hydra config
        generate_command = (
            f"python {self.config.fairseq_root}/fairseq_cli/generate.py "
            f" {self.config.data} "
            f" --path {model} "
            f" --task translation_multi_simple_epoch "
            f" --langs \"{','.join(self.config.langs.all)}\" "
            f'--lang-pairs "{src}-{tgt}"'
            f" --source-lang {src} "
            f" --target-lang {tgt} "
            f'--encoder-langtok "{self.config.encoder_langtok}"'
            " --decoder-langtok "
            f" --gen-subset {job_config.gen_split} "
            f" --beam {self.config.beam_size} "
            " --bpe 'sentencepiece' "
            f" --sentencepiece-model {self.config.spm_model} "
            " --sacrebleu "
            f" {'--fp16' if self.config.fp16 else ''}"
            f" {moe_params} "
            f" --max-sentences {max_sentences} "
            f" --results-path {out_dir}"
        )
        if self.config.get("debug", False):
            print(generate_command)
        else:
            subprocess.run(
                generate_command,
                shell=True,
                check=True,
            )
            subprocess.run(
                f"/bin/bash -o pipefail -c '"
                f"cat {out_dir}/generate-{job_config.gen_split}.txt"
                ' | grep -aP "^D-"'
                " | sort -nr -k1.2 "
                " | cut -f3' "
                f" > {out_dir}/gen_best.output",
                shell=True,
                check=True,
            )

            flores_split = "dev" if job_config.gen_split == "valid" else "devtest"
            ref_file = os.path.join(
                self.config.cluster.flores_path, flores_split, f"{tgt}.{flores_split}"
            )

            subprocess.run(
                f'lang="{job_config.lang}" SACREBLEU_FORMAT=text '
                f"sacrebleu -tok spm {ref_file} < {out_dir}/gen_best.output "
                f" > {out_dir}/bleu.results",
                shell=True,
                check=True,
            )


async def _main(config):
    launcher = hydra.utils.instantiate(config.launcher)
    module = GenerateMultiModule(config)
    await launcher.schedule(module)


@hydra.main(config_path="conf", config_name="generate_multi")
def main(config: DictConfig) -> None:
    asyncio.run(_main(config))


if __name__ == "__main__":
    main()
