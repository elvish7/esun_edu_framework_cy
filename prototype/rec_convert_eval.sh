# Function for evalute the rec file with the following two purposes: 
# (1) Convert the evaluate interaction to the '.rec' file (as the eval groundtruth)
# (2) Calculate the pre-defined metrics from the predictions

DATE=20181231
if [ -e rec/${DATE}.eval.truth.rec ]
then
    echo "TRUTH REC FILE EXISTED:  'rec/${DATE}.eval.truth.rec'"
else
    echo "TRUTH REC FILE CREATING: 'rec/${DATE}.eval.truth.rec'"

    python3 convert_to_rec.py \
        --path_transaction data/${DATE}/transaction_train.csv \
        --path_transaction_eval data/${DATE}/transaction_eval.csv \
        --path_user data/${DATE}/customer.csv \
        --path_item data/${DATE}/product.csv \
        --path_rec_output rec/${DATE}.eval.truth.rec
    echo "- START EVALUATE"
fi

echo -e

python3 rec_eval.py \
    -truth rec/${DATE}.eval.truth.rec \
    -pred rec/pred.rec \
    -metric 'P@5' \
    -metric 'R@5' \
    -metric 'mAP@5' \
    -metric 'mRR@5'
    # -pred rec/${DATE}.eval.pred.rec \
