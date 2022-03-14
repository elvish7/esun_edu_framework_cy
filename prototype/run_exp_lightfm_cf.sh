# # training with lightfm on esun fund
DATE=20181231
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --model 'LightFM' \
    --model_path 'models/lightfm' \
    --model_type 'cf' \
    --model_hidden_dimension 64 \
    --model_max_neg_sample 10 \
    --model_loss 'warp' \
    --training_do_evaluation \
    --training_verbose \
    --training_num_epochs 8 \
    --training_eval_per_epochs 3 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'precision@5' \
    --evaluation_metrics 'recall@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --use_heuristic 'frequency' \
    --use_heuristic 'volume' \
    --evaluation_results_csv results/lightfm_cf_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/lightfm_cf_rec_detail_${DATE}.tsv \
        > logs/lightfm_cf_exp_${DATE}.log & 
    # --use_heuristic 'random' \

DATE=20190630
nohup python3 train.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --model 'LightFM' \
    --model_path 'models/lightfm' \
    --model_type 'cf' \
    --model_hidden_dimension 64 \
    --model_max_neg_sample 10 \
    --model_loss 'warp' \
    --training_do_evaluation \
    --training_verbose \
    --training_num_epochs 8 \
    --training_eval_per_epochs 3 \
    --evaluation_diff \
    --evaluation_regular \
    --evaluation_metrics 'precision@5' \
    --evaluation_metrics 'recall@5' \
    --evaluation_metrics 'mAP@5' \
    --evaluation_metrics 'mRR@5' \
    --use_heuristic 'frequency' \
    --use_heuristic 'volume' \
    --evaluation_results_csv results/lightfm_cf_evaluation_${DATE}.csv \
    --evaluation_rec_detail_report results/lightfm_cf_rec_detail_${DATE}.tsv \
        > logs/lightfm_cf_exp_${DATE}.log & 
    # --use_heuristic 'random' \
