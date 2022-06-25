# NLLB-200 Training Scripts

These are scripts for training the NLLB-200 model and other variants. 

## Final NLLB-200 model
All of the scripts have `-c job` to print the configuration before running. Uncomment this to launch training. 

First, set the output directory for checkpoints and training logs:

`export OUTPUT_DIR=...`

NLLB-200 with Phase-Based Curriculum Learning:

`bash examples/nllb/modeling/train/scripts/run_flores200_4.4_final.sh $OUTPUT_DIR`

| update step | max_updates | lang_pairs_file | restore_file | reset_dataloader |
| - | - | - | - | - |
| 0 | 170000 | final_lang_pairs_cl3.txt | ~ | false |
| 170000   | 230000 | final_lang_pairs_cl2.txt | `$OUTPUT_DIR/checkpoint_17_170000.pt` | false |
| 230000   | 270000 | final_lang_pairs_cl1.txt | `$OUTPUT_DIR/checkpoint_22_230000.pt` | false |
| 270000   | 300000 | lang_pairs.txt | `$OUTPUT_DIR/checkpoint_25_270000.pt` | true |

## Other variants

### 1. Expert Output Masking
Without Curriculum Learning, overall_drop=0.3, expert_output_mask=0.2:

`bash examples/nllb/modeling/train/scripts/run_flores200_4.4_eom_sweep.sh $OUTPUT_DIR 0.3 0.2`

With Naive Curriculum Learning, overall_drop=0.3, expert_output_mask=0.2:

`bash examples/nllb/modeling/train/scripts/run_flores200_4.4_eom.sh $OUTPUT_DIR 0.3 0.2`

| update step | max_updates | lang_pairs_file | restore_file |
| - | - | - | - |
| 0 | 200000 | cl1_lang_pairs.txt | ~ |
| 200000   | 300000 | lang_pairs.txt | `$OUTPUT_DIR/checkpoint_27_200000.pt` |

### 2. Conditional MoE Routing
Without Curriculum Learning, overall_drop=0.3, conditional_gate_drop=0.2:

`bash examples/nllb/modeling/train/scripts/run_flores200_4.4_cmr_sweep.sh $OUTPUT_DIR 0.3 0.2`

With Naive Curriculum Learning, overall_drop=0.3, conditional_gate_drop=0.2:

`bash examples/nllb/modeling/train/scripts/run_flores200_4.4_cmr.sh $OUTPUT_DIR 0.3 0.2`

| update step | max_updates | lang_pairs_file | restore_file |
| - | - | - | - |
| 0 | 200000 | cl1_lang_pairs.txt | ~ |
| 200000   | 300000 | lang_pairs.txt | `$OUTPUT_DIR/checkpoint_27_200000.pt` |
