DIR=results

for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
    do 
    for s in 18 6 
        do python3 graph_embedding_main_wc.py --date ${d} --train_span ${s} --model tpr | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/tpr_warm.txt  
#    do python3 graph_embedding_main_wc.py --date ${d} --train_span ${s} --model smore --mode warp | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/smore_warp_warm.txt
    # do python3 graph_embedding_main_wc.py --date ${d} --train_span ${s} --model smore --mode bpr | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/smore_bpr_warm.txt
    done
done

