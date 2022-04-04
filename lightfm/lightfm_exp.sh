DIR=results
## Evaluation span: 1 month
len=1m
#for dim in 10 16 32 64 128 256
#do
#for e in 10 20 30 40 50
#do
## Pure CF no side information
for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do 
    for s in 18 6 
    do
#        python3 lightfm_main.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 10 | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_pure_cf_results.txt
        python3 lightfm_main.py --date ${d} --train_span ${s} --eval_duration ${len} --dim 128 --epoch 10 --item_ft --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_uifeatures_cf_results.txt
    done
# echo " " >> ${DIR}/lightfm_pure_cf_results.txt
done
# echo " " >> ${DIR}/lightfm_pure_cf_results.txt

### With User Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --dim ${dim} --epoch ${e} | grep 'Today' | awk -F' ' -vd=${dim} -ve=${e} '{print d,e,$2,$5}' >> ${DIR}/lightfm_pure_cf_results.txt
#done
#done
#echo " " >> ${DIR}/lightfm_pure_cf_results.txt
#done
### With User Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --user_ft  --dim 128 --epoch 10 | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_user_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_user_results.txt
### With Item Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --item_ft --dim 128 --epoch 10 | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_item_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_item_results.txt
### With User & Item Features
#for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
#do
#    python3 lightfm_main.py --date ${d} --eval_duration ${len} --user_ft --item_ft --dim 128 --epoch 10 | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/lightfm_cf_with_user_item_results.txt
#done
#echo " " >> ${DIR}/lightfm_cf_with_user_item_results.txt
