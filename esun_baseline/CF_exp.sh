
DIR=results
echo " " >> ${DIR}/CF_results.txt
len=1m
for d in 2019-06-30 #2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
do
    for span in 18 #12 6 3 1
    do
        rm data/old_cust_baseline/dataloader/tmp/*
        rm data/old_cust_baseline/preprocess/tmp/*
        python3 CF_baseline.py \
            --training_span ${span} \
            --date ${d} \
            --eval_duration ${len} | grep 'Today' | awk -F' ' '{print $2,$4,$6}' \
                >> ${DIR}/CF_results.txt
    done
done
