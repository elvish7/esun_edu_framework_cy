DIR=results
## Evaluation span: 1 month
len=1m

for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do 
    ## Pure CF no side information
    # python local_lightfm_main.py --date ${d} | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_pure_cf.txt
    ## Item features
    # python local_lightfm_main.py --date ${d} --item_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_itemft.txt
    ## User features
    # python local_lightfm_main.py --date ${d} --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_userft.txt
    ## Fake features
    python local_lightfm_main.py --date ${d} --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_fakeft.txt
    ### With User & Item Features
    # python local_lightfm_main.py --date ${d} --item_ft --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_uifeatures.txt
done

# item feature select
# for i in 1 2 3 4 5 6 7 8 
# do
#     python local_lightfm_main.py --item_ft --feature_i ${i} | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14,$17}' >> ${DIR}/lightfm_itemft_sel.txt
# done
# python3 lightfm_main.py --date 2018-12-31 --train_span 6 --eval_duration 1m --dim 128 --epoch 10 --item_ft --user_ft | grep 'Today' | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/lightfm_uifeatures_cf_results.txt