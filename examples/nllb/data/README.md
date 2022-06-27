# Data

## Primary Datasets

### Public data

The script `download_parallel_corpora.py` is provided for convenience
to automate download of many publicly available sources of MT data
which were used to train NLLB models. You should provide a parent
directory into which to save the data. Usage is as follows:

``` bash
python download_parallel_corpora.py --directory $DOWNLOAD_DIRECTORY
```

Note that there are a number of other adhoc datasets for which we are
not able to automate this process because they require an account or login
of some kind:

1. Chichewa News (https://zenodo.org/record/4315018#.YaTuk_HMJDY)
2. GELR (Ewe-Eng) (https://www.kaggle.com/yvicherita/ewe-language-corpus)
3. Lorelei (https://catalog.ldc.upenn.edu/LDC2021T02)

Important note on JW300 (described in https://aclanthology.org/P19-1310/):
at the time of final publication, the JW300 corpus was no longer publicly
available for MT training because of licensing isses with the Jehovah's
Witnesses organization, though it had already been used for the NLLB project.
We hope that it may soon be made available again.

### NLLB-Seed Data

NLLB-Seed datasets can be downloaded from [Flores-200 repo]().


## Mined Data

Mined metadata is available in [mining metadata](). [TODO].

## Backtranslated Data

Backtranslation data for NMT models can be generated using the script. [TODO].
