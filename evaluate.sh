
DATA_DIR=models/m6_moe_eom_4_xxl/
DATA_BIN=${DATA_DIR}data_bin
LANGS='fas,zho_Hans,rus,kor,ara_Arab,eng'
LANG_PAIRS='fas-eng,zho_Hans-eng,rus-eng,kor-eng,ara_Arab-eng,eng-fas,eng-zho_Hans,eng-rus,eng-kor,eng-ara_Arab'
SAVE_PATH=./models/m6_moe_eom_4_xxl/
EXPERT_NUM=4
RANDOM_PORT=175$(( $RANDOM % 50 + 1 ))

SRCS='fas,zho_Hans,rus,kor,ara_Arab'
tgt=eng
mkdir -p ${SAVE_PATH}/results
replication_count=$[ 32 / ${EXPERT_NUM} ]


for src in ${SRCS//,/ }; do
    echo predict $src to $tgt
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}


    # cat ${FSRC} | \
    fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
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
        --no-progress-bar |\
        tail -n 1 >  $FOUT.bleu
    cat ${FOUT}.bleu
done

TGTS='fas,zho_Hans,rus,kor,ara_Arab'
src=eng
for tgt in ${TGTS//,/ }; do
    echo predict $src to $tgt
    FSRC=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${src}
    FTGT=${DATA_DIR}/retrieved_data/test.${src}-${tgt}.${tgt}
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}

    fairseq-generate ${DATA_BIN} --path $SAVE_PATH/checkpoint_best.pt \
        --langs ${LANGS} \
        --lang-pairs ${LANG_PAIRS} \
        --task translation_multi_simple_epoch \
        --is-moe \
        --bpe "sentencepiece" \
        --sacrebleu \
        --encoder-langtok tgt --decoder-langtok \
        --sentencepiece-model ${DATA_DIR}/vocab_bin/sentencepiece.source.64000.model \
        --source-lang ${src} --target-lang ${tgt} \
        --distributed-world-size 32 --distributed-port ${RANDOM_PORT} \
        --batch-size 100 \
        --model-overrides "{'world_size': 32, 'moe_eval_capacity_token_fraction': 1.0, 'use_moe_pad_mask': False, 'pass_tokens_transformer_layer': False, 'replication_count': ${replication_count}}" \
        --no-progress-bar |\
        tail -n 1 >  $FOUT.bleu
    cat ${FOUT}.bleu
done


# Print
SRCS='fas,zho_Hans,rus,kor,ara_Arab'
tgt=eng
for src in ${SRCS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${tgt}-${src}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 7
    # cat $FOUT.chrf | grep score
done


TGTS='fas,zho_Hans,rus,kor,ara_Arab'
src=eng
for tgt in ${TGTS//,/ }; do
    FOUT=${SAVE_PATH}/results/predict.${src}-${tgt}.${tgt}
    echo -------------------
    echo ${src}-${tgt}
    cat $FOUT.bleu | cut -d ' ' -f 7
    # cat $FOUT.chrf | grep score
done