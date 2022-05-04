DIR=results
## Evaluation span: 1 month
len=1m

# feature selection

# item feature select
for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do
    for s in 18 6 
    do
        python3 lightfm_main_fakeft.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 20 --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_fakeft.txt
    done
done