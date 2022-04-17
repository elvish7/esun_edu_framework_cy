DIR=results
## Evaluation span: 1 month
len=1m

for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do 
    for s in 18 6 
    do
        ## Pure CF no side information
#        python3 lightfm_main.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 10 | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_pure_cf_results.txt
        ### With User & Item Features
        python3 lightfm_main.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 10 --item_ft --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_uifeatures_cf_results.txt
    done
done

# python3 lightfm_main.py --date 2018-12-31 --train_span 6 --eval_duration 1m --dim 128 --epoch 10 --item_ft --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_uifeatures_cf_results.txt