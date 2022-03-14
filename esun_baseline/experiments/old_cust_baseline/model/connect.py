item_matrix = calculate_similarity(item_feature)
cust_item_rating_matrix, user2nid, item2nid = matrix_mul_advance(
    cust_txn_rating, 
    item_matrix
)
