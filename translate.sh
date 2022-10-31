INPUT=${1}
OUTPUT=${2}
src=${3}
tgt=${4}
MODEL_PATH=${5:-/brtx/601-nvme1/haoranxu/BETTER_MMT/models/m6_moe_eom_4_xxl/}


DATA_DIR=${MODEL_PATH}
DATA_BIN=${DATA_DIR}data_bin
LANGS='fas,zho_Hans,rus,kor,ara_Arab,eng'
LANG_PAIRS='fas-eng,zho_Hans-eng,rus-eng,kor-eng,ara_Arab-eng,eng-fas,eng-zho_Hans,eng-rus,eng-kor,eng-ara_Arab'
SAVE_PATH=${DATA_DIR}
EXPERT_NUM=4
RANDOM_PORT=175$(( $RANDOM % 50 + 1 ))
replication_count=$[ 32 / ${EXPERT_NUM} ]

mkdir tmp_workplace
python scripts/spm_encode.py \
--model  ${DATA_DIR}/vocab_bin/sentencepiece.source.64000.model \
--input  ${INPUT} \
--outputs tmp_workplace/test.${src}-${tgt}.${src}

cp tmp_workplace/test.${src}-${tgt}.${src} tmp_workplace/test.${src}-${tgt}.${tgt}

fairseq-preprocess --task "translation" --source-lang ${src} --target-lang ${tgt} \
--trainpref tmp_workplace/test.${src}-${tgt} --validpref tmp_workplace/test.${src}-${tgt}  --testpref tmp_workplace/test.${src}-${tgt}  \
--destdir  tmp_workplace/data-bin/ --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
--srcdict ${DATA_BIN}/dict.eng.txt --tgtdict ${DATA_BIN}/dict.eng.txt 

fairseq-generate  tmp_workplace/data-bin/ --path $SAVE_PATH/checkpoint_best.pt \
    --langs ${LANGS} \
    --lang-pairs ${LANG_PAIRS} \
    --task translation_multi_simple_epoch \
    --is-moe \
    --sacrebleu \
    --encoder-langtok tgt --decoder-langtok \
    --bpe "sentencepiece" \
    --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.64000.model \
    --source-lang ${src} --target-lang ${tgt} \
    --distributed-world-size 32 --distributed-port ${RANDOM_PORT} \
    --batch-size 100  \
    --model-overrides "{'world_size': 32, 'moe_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
    --no-progress-bar  > tmp_workplace/tmp.output

cat tmp_workplace/tmp.output | sort -t '-' -nk 2 | grep -P "^D-" | cut -f 3- > ${OUTPUT}

rm -rf tmp_workplace