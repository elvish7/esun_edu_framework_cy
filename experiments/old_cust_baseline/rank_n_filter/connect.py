#txn_dig = extract_dig_channel_purchase(w103, dig_start_dt, dig_end_dt)
#('txn_dig.feather')
dig_cnt = get_dig_cnt(txn_dig)

# Break Point 1: cust_item_rating_matrix, user2nid, item2nid
cust_item_rank_matrix_1 = convert_rating_to_rank(cust_item_rating_matrix)
# Break Point 2: cust_item_rating_matrix, cust_item_rank_matrix_1, user2nid, item2nid
cust_item_rank_matrix_2 = re_rank_by_hotness_n_novalty(
    cust_item_rank_matrix_1, 
    user2nid, 
    item2nid, 
    dig_cnt, 
    cust_txn_exclude
)
# Break Point 3: cust_item_rating_matrix, cust_item_rank_matrix_2, user2nid, item2nid
_cust_item_rank_matrix, item2nid_3 = filter_by_items(
    cust_item_rank_matrix_2, item2nid, featured_funds
)

cust_item_rank_matrix_out = re_rank(_cust_item_rank_matrix) 

cust_item_rating_matrix_out, item2nid_out = filter_by_items(
    cust_item_rating_matrix, item2nid, featured_funds
)
# Break Point 4: cust_item_rating_matrix_out, cust_item_rank_matrix_out, user2nid, item2nid_out

