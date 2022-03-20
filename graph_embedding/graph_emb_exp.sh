DIR=results
for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
    do 
    for s in 18 6 
#    do python3 graph_embedding_main.py --date ${d} --train_span ${s} --model tpr | grep "Today" | awk -F' ' '{print $2,$4,$10,$14}' >> ${DIR}/post_midterm_graph_embedding_tpr.txt  
    do python3 graph_embedding_main_wc.py --date ${d} --train_span ${s} --model smore | grep "Today" | awk -F' ' '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/graph_embedding_tpr_warm.txt
    done
done
