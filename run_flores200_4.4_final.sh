drop=$1
moe_freq=4
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=/data/nllb/users/$USER/models/flores200.full.v4.4/ \
    cfg.max_update_str="mu170.230.270.mf${moe_freq}" \
    cfg.max_updates=170000 \
    cfg.dataset.lang_pairs_file=examples/nllb/modeling/scripts/flores200/final_lang_pairs_cl3.txt \
    cfg.lr=0.002 \
    cfg.resume_finished=true \
    cfg.dropout=${drop} \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq ${moe_freq} --moe-eom $2 " \