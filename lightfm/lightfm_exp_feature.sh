DIR=results
## Evaluation span: 1 month
len=1m

# feature selection
for fn in 4 5 6 7 8 9 #2 3 4 5 6 7 8 9
do
    for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
    do 
        for s in 18 6 
        do
            ### With User & Item Features
            python3 lightfm_main.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 10 --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_userfeature_select_results${fn}.txt
        done
    done
done
