# bash run_sweep $drop $moe_tok_drp

drop=$1
moe_freq=4
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=/data/nllb/users/$USER/checkpoint/flores200.full.v4.4/sweep.moe_drop/ \
    cfg.max_update_str="demo" \
    cfg.max_updates=1 \
    cfg.lr=0.002 \
    cfg.resume_finished=true \
    cfg.dropout=${drop} \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq ${moe_freq} --moe-tok-drp $2 " \
    cfg.restore_file="/data/nllb/users/shru/checkpoint/flores200.full.v4.4/sweep.moe_drop/moe128.mu2.mf${moe_freq}.uf1.tmp1.lr0.002.drop0.3.maxtok4096.seed2.max_pos512.shem.NBF.moe_w0.01.all.mtdrp0.2.entsrc.det.ELS24.DLS24.encffnx8192.decffnx8192.E2048.H16.ATTDRP0.1.RELDRP0.0.ngpu256/checkpoint_12_200000.pt" \
    cfg.no_save=true \
    cfg.validate_interval_updates=1 \
    cfg.log_interval=1 \
    cfg.reset_dataloader=true \
    cfg.eval_lang_pairs="/shared/home/shru/projects/fairseq-py/examples/nllb/modeling/scripts/flores200/non_english_lang_pairs.txt" \
    # cfg.train_subset=\"train,train_mining\" \

