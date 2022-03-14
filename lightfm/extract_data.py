import pandas as pd
from db_connection.utils import get_conn
from utils import load_cust_pop_all, load_w106_all, load_w103_all, load_w103_eval
from dateutil.relativedelta import relativedelta

def get_data_dt(etl_time_string, backward_months):
	today = pd.to_datetime(etl_time_string, format='%Y-%m-%d')
	data_start_dt = today + relativedelta(days=1)
	data_1m_end_dt = today + relativedelta(months=backward_months)
	data_7d_end_dt = today + relativedelta(days=7)
	data_start = data_start_dt.strftime('%Y-%m-%d')
	data_1m_end = data_1m_end_dt.strftime('%Y-%m-%d')
	data_7d_end = data_7d_end_dt.strftime('%Y-%m-%d')
	return data_start, data_1m_end, data_7d_end

span=18
rawdata_conn = get_conn('edu')

for today in ['2018-12-31', '2019-06-30']:
	customer_df=load_cust_pop_all(
		today, rawdata_conn
	)
	print(customer_df)
	today_folder = today.replace("-", "")
	customer_df.to_csv(f"{today_folder}/customer.csv")

	product_df = load_w106_all(
		rawdata_conn
	)
	product_df.to_csv(f"{today_folder}/product.csv")
	print(product_df)

	transaction = load_w103_all(
		today, rawdata_conn, span
	)
	transaction.to_csv(f"{today_folder}/transction_train.csv")
	print(transaction)

	day_1, month_1, day_7 = get_data_dt(today, 1)

	transaction_eval = load_w103_eval(
		today, rawdata_conn, day_1, month_1
	)
	transaction_eval.to_csv(f"{today_folder}/transaction_eval.csv")
	print(transaction_eval)

	day_1, month_3, day_7 = get_data_dt(today, 3)

	transaction_test = load_w103_eval(
		today, rawdata_conn, day_1, month_3
	)
	transaction_eval.to_csv(f"{today_folder}/transaction_test.csv")
	print(transaction_test)
