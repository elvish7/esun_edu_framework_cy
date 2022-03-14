w106 = load_w106(rawdata_conn=rawdata_conn)

w107 = load_w107(rawdata_conn=rawdata_conn)

available_funds = load_available_funds(rawdata_conn=rawdata_conn)

_cust_item_rank_matrix, _item2nid = filter_by_items(
    cust_item_rank_matrix, item2nid, available_funds
)

_cust_item_rank_matrix = re_rank(_cust_item_rank_matrix) 

_cust_item_rating_matrix, _item2nid = filter_by_items(
    cust_item_rating_matrix, item2nid, available_funds
)

# from dev.etl3.process_utils.risk_management import reorganize_by_RR
# from dev.etl3.process_utils.risk_management import select_by_cust_risk

recmd = reorganize_by_RR( 
    _cust_item_rank_matrix,
    _cust_item_rating_matrix,  
    user2nid, 
    _item2nid, 
    w106, 
    K = 5
)

output = select_by_cust_risk(recmd, w107, w106)
