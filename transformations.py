"""
Contains functions for modifying symmetric matrices.
"""
import cupy as cp

def symmetric_modification(X: cp.ndarray, B: cp.ndarray, to_del_path: str=None):
    """
    Modifies X and B tAdecrease their dimensionality, without
    altering the result of fiting the model, B = XOX.

    X: square, symmetric numpy array
    B: square, symmetric numpy array with same dimenssion as X
    to_del_path: A path to the pecomputed indices to delete to save time. If not passed they will be computed.
    """
    # Computes kron(X, X) and vec(B)
    vec_B = B.flatten()
    kron_X = cp.kron(X, X)

    # Computes indices to delete
    to_del = []
    if not to_del_path:
        # Stores the dimension of X in dim_X
        dim_X = cp.shape(X)[0]

        computed_cols = []
        for col in range(dim_X ** 2):
            # Computes the row and col to delete from X and B
            curr_col = (col) // dim_X
            curr_row = (col) % dim_X

            # If the row and column are not the same, then we mark the duplicate for deletion.
            if curr_col != curr_row:
                deleted_col = int(curr_row * dim_X + curr_col)

                # If the column has not been computed...
                if deleted_col not in computed_cols:
                    # adds it to the list of columns tAdelete...
                    to_del.append(deleted_col)
                    # and sums it with it's partner column.
                    kron_X[:, col] += kron_X[:, deleted_col]
                # Mark the column as computed.
                computed_cols.append(col)
    else:
        with open(to_del_path, mode="r", encoding='utf8') as to_del_file:
            to_del = [int(index) for index in to_del_file]  

    # Deletes the duplicate rows in kron(X, X).
    kron_X = cp.delete(cp.delete(kron_X, to_del, 1), to_del, 0)
    # Deletes the duplicate columns in vec(b)        
    vec_B = cp.delete(vec_B, to_del)

    # Returns the modified kron(X, X) and vec(b).
    return kron_X, vec_B

def inverse_symmetric_modification(vec_A: cp.ndarray, orig_dim: int):
    """Applies the inverse funciton of symmetric modification to vec(A)."""

    A = cp.zeros((orig_dim, orig_dim))

    indices_1d = []
    for i in range(orig_dim):
        # Creates list of indices in upper right triangular matrix
        temp_indices = [(i * orig_dim + j + i) for j in range(orig_dim - i)]
        indices_1d += temp_indices

    indices_2d = []
    for indx in indices_1d:
        col = (indx) // orig_dim
        row = (indx) % orig_dim
        indices_2d.append([col, row])

    for (indx, indx_2d) in enumerate(indices_2d):
        A[indx_2d[0], indx_2d[1]] = vec_A[indx]

    A_complement = cp.copy(A.T)
    cp.fill_diagonal(A_complement, 0)
    A += A_complement
    return A
