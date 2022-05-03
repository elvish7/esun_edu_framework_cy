DIR=results

# for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
#    do 
#    for s in 18 6 
#    do python local_tpr_main.py --date ${d} --train_span ${s} --model tpr_notext | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/local_tpr_tune.txt  
# #    do python graph_embedding_main_wc.py --date ${d} --train_span ${s} --model smore --mode warp | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/smore_warp_warm.txt
# #    do python graph_embedding_main_wc.py --date ${d} --train_span ${s} --model smore --mode bpr | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/smore_bpr_warm.txt
#    done
# done

# tunning
for dim in 128 64 256
do 
    for neg in 5 10 20 2 
    do
        for lr in 0.025 0.001 0.01 0.05 0.1
        do
            for t in 100 10 2
            do
                python local_tpr_main.py --dim ${dim} --neg ${neg} --l ${lr} --t ${t} --model tpr_notext | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14,$16}' >> ${DIR}/local_tpr_tune.txt  
            done
        done
    done
done

# feature selection
# for fn in 4 5 6 7 8 9
# do
#     for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
#         do 
#         for s in 18 6 
#         do 
#         python3 graph_embedding_main_wc.py --date ${d} --train_span ${s} --model tpr --feature_number ${fn} | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/tpr_warm_f${fn}.txt  
#         done
#     done
# done
