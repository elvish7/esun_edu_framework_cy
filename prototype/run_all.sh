# # training with lightfm on esun fund
EPOCHS=10
EMBED_SIZE=64
DATE=$1

# LighFM CF
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --model 'LightFM' \
    --model_path 'models/lightfm' \
    --model_type 'cf' \
    --model_hidden_dimension ${EMBED_SIZE} \
    --model_max_neg_sample 100 \
    --model_loss 'warp' \
    --training_do_evaluation \
    --training_verbose \
    --training_num_epochs ${EPOCHS} \
    --training_eval_per_epochs 1 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'P@5' \
    --evaluation_metrics 'R@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --use_heuristic 'frequency' \
    --use_heuristic 'volume' \
    --evaluation_results_csv results/lightfm_cf_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/lightfm_cf_rec_detail_${DATE}.tsv \
        > logs/lightfm_cf_exp_${DATE}.log & 

# LightFM hybrid
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --model 'LightFM' \
    --model_path 'models/lightfm' \
    --model_type 'hybrid' \
    --model_hidden_dimension ${EMBED_SIZE} \
    --model_max_neg_sample 100 \
    --model_loss 'warp' \
    --training_do_ \
    --training_verbose \
    --training_num_epochs ${EPOCHS} \
    --training_eval_per_epochs 1 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'P@5' \
    --evaluation_metrics 'R@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --evaluation_results_csv results/lightfm_bybrid_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/lightfm_hybrid_rec_detail_${DATE}.tsv \
        > logs/lightfm_hybrid_exp_${DATE}.log & 

# LighFM meta only
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --model 'LightFM' \
    --model_path 'models/lightfm' \
    --model_type 'cf' \
    --model_hidden_dimension ${EMBED_SIZE} \
    --model_max_neg_sample 100 \
    --model_loss 'warp' \
    --training_do_ \
    --training_verbose \
    --training_num_epochs ${EPOCHS} \
    --training_eval_per_epochs 1 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'P@5' \
    --evaluation_metrics 'R@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --evaluation_results_csv results/lightfm_meta_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/lightfm_meta_rec_detail_${DATE}.tsv \
        > logs/lightfm_meta_exp_${DATE}.log & 

# SMORe
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --path_temp_network temp/networks.${DATE}.txt \
    --path_temp_rep temp/smore_reps.${DATE}.txt \
    --model 'SMORe' \
    --model_path 'models/smore' \
    --model_hidden_dimension ${EMBED_SIZE} \
    --model_max_neg_sample 100 \
    --model_loss 'warp' \
    --training_do_ \
    --training_verbose \
    --training_num_epochs ${EPOCHS} \
    --training_eval_per_epochs 1 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'P@5' \
    --evaluation_metrics 'R@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --evaluation_results_csv results/smore_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/smore_rec_detail_${DATE}.tsv \
        > logs/smore_exp_${DATE}.log & 
