DIR=results
## Evaluation span: 1 month
for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do
    for span in 18 6
    do
    python3 main.py --date ${d} --train_span ${span} | grep 'Today' | awk -F' '  '{print $2,$4,$6,$8,$10,$12,$14}' >> ${DIR}/warm_exp_popular.txt
    done
done
