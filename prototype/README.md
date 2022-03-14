<h1> Explainable Mutual Fund Recommendation  </h1>


## Data
Please see 'DATA_DESCRIPTION.md' for mode detail.

## Recommender System Methods
Baseline
*  Collabarative Fiiltering
*  PersonFreq
*  PersonVolume

Stable
*  LightFM Meta
*  LightFM PureCF   
*  LightFM Hybrid 

Advanced
*  DGL
*  GCN

## Part I: Fund Recommedation

### Training

##### Supported models
1. Heuristic
2. LightFM (CF/Hybrid/Meta)
3. SMORe 
```
# Process 3 models in parallel
bash run_all.sh <THE_DATE>
```

##### Arugments
You can also tune the detail parameter settings of each method in training pipeline. 
```
# Commonly used arguments 
--model <backbone model. e.g. 'LighFM' or 'SMORe'>
--model_type <LightFM model type. e.g.'CF', 'meta', 'hybrid' >
--model_hidden_dimension <latent embedding size. e.g. 16, 128 >
--evaluation_metrics <evaluation metrics, calculate for mean with the format metric@cutoff. e.g. "P@5", "mAP@5", "mRR@5", "R@5" >
--use_heuristic <suppored heuristic methods. e.g.'frequency', 'volume', 'random'>
```
For example, LightFM with pure-CF method
```
EPOCHS=10
EMBED_SIZE=64
DATE=20181231

python3 train.py \
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
       > logs/lightfm_cf_exp_${DATE}.log
```
For another example, SMORe
```
python3 train.py \
   --path_transaction data/${DATE}/transaction_train.csv \
   --path_transaction_eval data/${DATE}/transaction_eval.csv \
   --path_user data/${DATE}/customer.csv \
   --path_item data/${DATE}/product.csv \
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
       > logs/smore_exp_${DATE}.log
```
#### Evaluataion
To use the evaluation pipeline, you need a prediction rec file with the format like the example below:
```
# prediction rec file <UID>\t<PID>\t<TYPE>\t<REGULAR>\t<SCORE>\t<RANK>
CFDAXWccjJPoVInuiF0mMg==        AG25    EXPLOIT SOLO    0       2
CFDAXWccjJPoVInuiF0mMg==        XXXX    EXPLOIT SOLO    0       1
CFDAXWccjJPoVInuiF0mMg==        JJ15    EXPLOIT REGULAR 0       2
CFDAXWccjJPoVInuiF0mMg==        XXXX    EXPLOIT REGULAR 0       1
CFDAwH4y/ssuYSedFy8UMw==        CC89    EXPLOIT REGULAR 0       2
CFDAwH4y/ssuYSedFy8UMw==        XXXX    EXPLOIT REGULAR 0       1
CFDA9UDJnLAm4/0txbPMVQ==        AP06    EXPLORE NA      0       2
CFDA9UDJnLAm4/0txbPMVQ==        XXXX    EXPLORE NA      0       1
```

Later you could directly use the evaluate pipeline
```
bash rec_convert_eval.sh <DATE>
```
In the evaluation pipeline, you need to convert the ground truth interaction into '.rec' format. For xample.
```
# truth rec file <UID>\t<PID>\t<TYPE>\t<REGULAR>\t<RATING>
CFDAXWccjJPoVInuiF0mMg==        AG25    EXPLOIT SOLO    1.0     
CFDAXWccjJPoVInuiF0mMg==        JJ15    EXPLOIT REGULAR 1.0     
CFDAwH4y/ssuYSedFy8UMw==        CC89    EXPLOIT REGULAR 1.0     
CFDA9UDJnLAm4/0txbPMVQ==        AP06    EXPLORE NA      1.0                                      
```
Convert from the evaluation transaction (includes the preprocess pipeline) by the following code, which will save the corresponding rec file in the defined argument '--path_trainsaction_truth'
```
DATE=20181231
python3 convert_to_rec.py \
    --path_transaction data/${DATE}/transaction_train.csv \
    --path_transaction_eval data/${DATE}/transaction_eval.csv \
    --path_user data/${DATE}/customer.csv \
    --path_item data/${DATE}/product.csv \
    --path_transaction_truth rec/${DATE}.eval.truth.rec
```
And evaluate by the code "rec_eval.py"
```
DATE=20181231
python3 rec_eval.py \
   -truth rec/${DATE}.eval.truth.rec \ 
   -pred rec/pred.rec \     
   -metric 'P@5' \          
   -metric 'R@5' \          
   -metric 'mAP@5' \
   -metric 'mRR@5'
```
The results would be like 
```
TRUTH REC FILE EXISTED:  'rec/20181231.eval.truth.rec'

EvalDict({                
          SUBSET     USERS     EXAMPLES 
        * EXPLORE    2305      2826     
        * EXPLOIT    33355     62403    
        * REGULAR    31763     59054    
        * SOLO       2747      3349                     
})
==============================
 P@5     on EXPLORE    0.0001
 R@5     on EXPLORE    0.0004
 mAP@5   on EXPLORE    0.0004
 mRR@5   on EXPLORE    0.0004
 P@5     on EXPLOIT    0.0000
 R@5     on EXPLOIT    0.0001
 mAP@5   on EXPLOIT    0.0001
 mRR@5   on EXPLOIT    0.0001
 P@5     on REGULAR    0.0000
 R@5     on REGULAR    0.0001
 mAP@5   on REGULAR    0.0001
 mRR@5   on REGULAR    0.0001
 P@5     on SOLO       0.0001
 R@5     on SOLO       0.0004
 mAP@5   on SOLO       0.0004
 mRR@5   on SOLO       0.0004
==============================
```

#### Results

Methods  | P@5 | mAP@5 | mRR@5 | R@5
:------- |:---:|:-----:|:-----:|:--:
Collabarative Fiiltering  | -     | -     | -     |
PersonFreq                | -     | -     | -     |
PersonVolume              | -     | -     | -     |
||
LightFM Meta             | -     | -     | -     |   
LightFM PureCF           | -     | -     | -     |         
LightFM Hybrid           | 0.000 | 0.000 | 0.000 | 0.000 
||
DGL                      | -     | -     | -     |   
GCN                      | -     | -     | -     |         

<hr/>

## Fund Explanation
