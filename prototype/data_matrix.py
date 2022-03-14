"""
In fact, I dont think this matrix class is necessary. 
But for now, keep it consistent to lightfm's design.
"""
import logging
import numpy as np
import scipy.sparse as sp

class COOMatrix(object):
    def __init__(self, shape, dtype=None):
        """
        dtype: type of interaction (rating)
        """
        self.shape = shape
        self.dtype = np.int32 if dtype is None else dtype
        if (self.dtype == np.int32 or self.dtype == int):
            self.type_flag = "i"
        elif (self.dtype == np.float32 or self.dtype == float):
            self.type_flag = "f"

        # self.rows = array.array("i")
        # self.cols = array.array("i")
        # self.data = array.array(self.type_flag)
        self.rows = np.array([], dtype=np.int32)
        self.cols = np.array([], dtype=np.int32)
        self.data = np.array([], dtype=self.dtype)

    def append_all(self, all_i=None, all_j=None, all_v=None):
        if all_i is not None:
            self.rows = np.array(all_i).astype(np.int32)
        if all_j is not None:
            self.cols = np.array(all_j).astype(np.int32)
        if all_v is not None:
            self.data = np.array(all_v).astype(self.dtype)

    def append(self, i, j, v):
        self.rows = np.append(self.rows, i)
        self.cols = np.append(self.cols, j)
        self.data = np.append(self.data, v)

    def tocoo(self):
        assert (len(self.rows), len(self.cols)) != self.shape, \
                "The interaction data and matrix shape are inconsistent."
        return sp.coo_matrix((self.data, (self.rows, self.cols)), shape=self.shape)

    def __len__(self):
        return len(self.data)


class NETList(object):

    def __init__(self, shape, dtype=None):
        self.shape = shape

        self.dtype = np.int32 if dtype is None else dtype
        if (self.dtype == np.int32 or self.dtype == int):
            self.type_flag = "i"
        elif (self.dtype == np.float32 or self.dtype == float):
            self.type_flag = "f"

        self.user_col = np.array([], dtype=np.int32)
        self.item_col = np.array([], dtype=np.int32)
        self.data_col = np.array([], dtype=self.dtype)

    def append_all(self, all_i=None, all_j=None, all_v=None):
        if all_i is not None:
            self.user_col = np.array(all_i).astype(np.int32)
        if all_j is not None:
            self.item_col = np.array(all_j).astype(np.int32)
        if all_v is not None:
            self.data_col = np.array(all_v).astype(self.dtype)

    def totxt(self, filename='temp.txt'):
        f = open(filename, 'w')
        for i, (user_id, item_id, rating) in enumerate(
                zip(self.user_col, self.item_col, self.data_col)
        ):
            f.write(f"USER_{user_id} ITEM_{item_id} {rating}\n")

        print(f"{i} interaction built...into '{filename}'.")








