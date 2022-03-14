sparse_dense_setting_generator = create_sparse_dense_setting_generator(
    category_cols=category_cols,
    sparse_feat=sparse_feat,
    numeric_cols=numeric_cols, 
    dense_feat=dense_feat, 
    USE_CHID=USE_CHID
)

chids = load_chids(chid_file=chid_file)
sampled_chids = sample_chids(chids, n_sample = n_sample)
('sampled_chids.npy')
chid_to_nid_map = build_chid_to_nid_map(sampled_chids)
df_cdtx = load_cdtx(sampled_chids, cdtx_file = cdtx_file) 
df_cdtx = convert_uid_to_nid(df_cdtx, chid_to_nid_map)
df_cdtx = add_month(df_cdtx)
('df_cdtx.feather')
df_full_y_sum = make_chid_x_month_table(df_cdtx)
df_cdtx_monthly_objam = calculate_monthly_target(df_cdtx)
df_full_y_sum = merge_with_another_table(df_full_y_sum, df_cdtx_monthly_objam, join_method=LEFT)
df_cust_f = load_cust_f(sampled_chids, cust_f_file=cust_f_file)
df_cust_f = convert_uid_to_nid(df_cust_f, chid_to_nid_map)
('df_cust_f.feather')
df_full_y_sum = merge_with_another_table(df_full_y_sum, df_cust_f, join_method=INNER)
df_full_y_sum = add_mean_of_previous_two_months(df_full_y_sum)
df_full_y_sum = cast_time_column_to_np_datatime64(df_full_y_sum, time_column = time_column_data_dt)
df_cdtx = cast_time_column_to_np_datatime64(df_cdtx, time_column = time_column_csmdt)
df_cdtx = add_duration_since_20180101(df_cdtx, time_column = time_column_csmdt, result_column=result_column_timestamp_1)
df_cdtx = add_duration_since_last_trans(df_cdtx, time_column = time_column_csmdt, result_column=result_column_timestamp_0)
df_input, feature_map = extract_feature_cols_and_encode_categoricals(df_cdtx, numeric_cols=numeric_cols, category_cols=category_cols)
('df_input.feather', 'feature_map.npy')
df_feat_input, cust_feature_map = extract_feature_cols_and_encode_categoricals(df_cust_f, numeric_cols=cust_numeric_cols, category_cols=cust_category_cols)
('df_feat_input.feather', 'cust_feature_map.npy')
df_feat_input = cast_time_column_to_np_datatime64(df_feat_input, time_column = time_column_data_dt)
df_feat_input = add_duration_since_20180101(df_feat_input, time_column = time_column_data_dt, result_column=result_column_timestamp)
('df_feat_input.2.feather')
df_y = extract_target_columns(df_full_y_sum, target_cols=target_cols)

df_y = add_duration_since_20180101(
    df_y, 
    time_column = time_column_data_dt, 
    result_column=result_column_timestamp
)
('df_y.feather')
x_train, x_test, f_train, f_test, y_train, y_test = split_data(
    df_input, 
    df_feat_input, 
    df_y, 
    window_size = window_size, 
    test_size = test_size
)
('x_train.npy', 'x_test.npy', 'f_train.npy', 'f_test.npy', 'y_train.npy', 'y_test.npy')
y_train, y_test, y_columns = add_objam_mean_M3_diff_as_new_target(
    df_y, 
    y_train, 
    y_test
)
('y_train.npy', 'y_test.npy', 'y_columns.npy')
columns = extract_x_f_y_columns(df_input, df_feat_input, y_columns)
sparse_dims, sparse_index, dense_dims, dense_index = sparse_dense_setting_generating_process(sparse_dense_setting_generator, feature_map, chid_to_nid_map)
x_train_sparse, x_train_dense, x_test_sparse, x_test_dense = ProcessX.process(x_train, x_test, sparse_index, dense_index)
('x_train_sparse.npy', 'x_train_dense.npy', 'x_test_sparse.npy', 'x_test_dense.npy')
train_objmean, train_tscnt, train_label_0, test_objmean, test_tscnt, test_label_0 = ProcessY.process(y_train, y_test, columns)
('train_objmean.npy', 'train_tscnt.npy', 'train_label_0.npy', 'test_objmean.npy', 'test_tscnt.npy', 'test_label_0.npy')
train_dataset = build_TensorDataset(x_train_dense, x_train_sparse, train_objmean, train_tscnt, train_label_0)
test_dataset = build_TensorDataset(x_test_dense, x_test_sparse, test_objmean, test_tscnt, test_label_0)