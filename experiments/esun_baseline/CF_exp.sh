
DIR=results
echo " " >> ${DIR}/CF_results.txt
len=1m
for d in 2019-06-30 2019-05-31 2019-04-30 2019-03-31 2019-02-28 2019-01-31 2018-12-31
do
    python3 CF_baseline.py --date ${d} --eval_duration ${len} | grep 'Today' | awk -F' ' '{print $2,$5}' >> ${DIR}/CF_results.txt
done
