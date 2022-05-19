# bash run_sweep $drop $moe_tok_drp

fairseq="/home/annaysun/rsc/workspace/fairseq-py2"
cd $fairseq
drop=$1
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.resume_finished=false \
    cfg.output_dir=/checkpoint/nllb/users/$USER/checkpoint/flores200.full.v4.4/sweep.moe_drop/ \
    cfg.max_updates=100000 \
    cfg.dataset.eval_lang_pairs_file=examples/nllb/modeling/scripts/flores200/eval_lang_pairs_eng400_noneng20.txt \
    cfg.dropout=${drop} \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq 2 --moe-tok-drp $2 "
    # cfg.model_type.moe_param=" --moe --moe-freq 2 --moe-clsr --clsr-gt-drp $2 "
#-c job
