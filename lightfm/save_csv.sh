## Evaluation span: 1 month
len=1m

for d in 2018-12-31 2019-01-31 2019-02-28 2019-03-31 2019-04-30 2019-05-31 2019-06-30
do 
    for s in 18 6 
    do
        python3 save_csv.py --date ${d} --train_span ${s} 
    done
done

