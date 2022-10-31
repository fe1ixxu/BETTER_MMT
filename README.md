## Environment:
```
conda create -n better-moe python=3.8
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout origin/experts_lt_gpus_moe_reload_fix
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e ./
```

## Translate from xxx<->eng
To translate from any languauge from or to English, run the command:
```
bash translate.sh ${INPUT} ${OUTPUT} ${SRC_LANG} ${TGT_LANG} ${MODEL_PATH}
```
Five inputs should be:
* `${INPUT}`: input file that is needed to be translated
* `${OUTPUT}`: output file
* `${SRC_LANG}`: source language id 
* `${TGT_LANG}`: target language id
* `${MODEL_PATH}`: Model path, now located at `/brtx/601-nvme1/haoranxu/BETTER_MMT/models/m6_moe_eom_4_xxl/`

**Note that language id only supports languages that used in BETTER**:
* fas: Farsi
* zho_Hans: Chinese (Simplified)
* rus: Russian
* kor: Korean
* ara_Arab: Arabic
* eng: English
