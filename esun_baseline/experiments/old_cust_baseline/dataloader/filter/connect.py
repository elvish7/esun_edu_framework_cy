# w103 = load_w103(txn_start_dt, txn_end_dt, rawdata_conn=rawdata_conn)

w107 = load_w107(rawdata_conn=rawdata_conn)

last_6m_new_cust = load_last_6m_new_cust(
            new_cust_start_dt, 
            new_cust_end_dt, 
            rawdata_conn=rawdata_conn
            )

cust_txn_exclude = load_cust_txn_exclude(
            txn_exclude_start_dt, 
            txn_exclude_end_dt, 
            rawdata_conn=rawdata_conn
            )

featured_funds = load_featured_funds(rawdata_conn=rawdata_conn)
available_funds = load_available_funds(rawdata_conn=rawdata_conn)