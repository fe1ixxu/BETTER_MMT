# Helper scripts for NLLB Modeling

## `generate_backtranslations.sh`

This script takes a binarized, sharded monolingual corpus and backtranslates it. The script will look for the binarized monolingual corpus under `${MONODIR}/data_bin/shard{000,001,...}/*.{bin,idx}`. We split the monolingual corpus into shards as jobs on slurm may fail or be pre-empted, and this allows us to restart individual shards when it they fail rather than having to redo the whole corpus.

The script will take care of re-running any shards that are incomplete – all you have to do is run it again with the same arguments. To detect whether a shard was not fully backtranslated, it will compare the number of lines in the output with the number of lines in the input (using the utility script `count_idx.py`). If the translated lines are >95% of the input lines, the shard is considered complete. Otherwise, the incomplete outputs will be deleted and the shard will be re-run. *NB*: Make sure all backtranslation jobs are complete before re-running the script with the same arguments – jobs that are still running will otherwise be mistaken by the script as having failed.

Before running the script please make sure you populate these variables in the script:
* `$MODEL`: The path to the model checkpoint file.
* `$MODEL_LANGS`: The list of languages supported by the model, in the same order they were passed during training on fairseq.
* `$MONODIR`: The path to the monolingual data directory.
* `$OUTDIR`: The path to the output directory where you want to store the generated backtranslations.


Example command:
```
examples/nllb/modeling/scripts/generate_backtranslations.sh x_en awa_Deva,fra_Latn,rus_Cyrl
```

Parameters:
* The first argument is whether the backtranslation generation is from or out of english, `en_x` or `x_en` respectively.
* The second argument is the list of languages to generate backtranslations for, a comma separated string of lang names. Eg: `awa_Deva,fra_Latn,rus_Cyrl`

## `count_idx.py`

This is a utility script to count the number of sentences in a fairseq binarized dataset (`.idx` file).
It's used by `generate_backtranslations.sh` to figure out which shards of a corpus have been fully backtranslated.
