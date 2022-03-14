cleaned_w106 = fix_empty_string_on_column(
    w106, 
    column = 'mkt_rbot_ctg_ic')

cleaned_w106 = rename_column(
    cleaned_w106, 
    columns = {'wm_prod_code':'item_id'}
)

item_attr = extract_cat_attrs(
                cleaned_w106, 
                id_name='item_id',
                attr_cols = [
                    'mkt_rbot_ctg_ic',
                    'prod_detail_type_code',
                    'prod_ccy',
                    'prod_risk_code']
                )

item_nav = encode_net_average_price(w118)

txn_dig = extract_dig_channel_purchase(w103, dig_start_dt, dig_end_dt)
('txn_dig.feather')
item_ranking = extract_purchase_amount_rank(txn_dig, max_rank = 50)

txn_matrix = obtain_user_item_matrix(w103, drop_duplicate=True)

item_feature = merge_fix_features(
    txn_matrix, 
    item_attr, 
    item_ranking, 
    item_nav
)
('item_feature.feather')
cust_txn_rating = normalize_amt_by_cust(cust_txn_amt)
('cust_txn_rating.feather')
