"""
Contains functions for modifying symmetric matrices.
"""
import cupy as cp

def symmetric_modification(X: cp.ndarray, B: cp.ndarray, to_del_path: str=None):
    """
    Modifies X and B to decrease their dimensionality, without
    altering the result of fitting the model, B = XOX.

    X: square, symmetric numpy array
    B: square, symmetric numpy array with same dimension as X
    to_del_path: A path to the precomputed indices to delete to save time. If not passed they will be computed.
    """
    # The symmetric matrix equation B = X ⊗ X · vec(A) has redundant columns
    # in kron(X, X) because X is symmetric: entry (i,j) equals (j,i).
    # We exploit this by summing each pair of mirror columns and deleting the
    # duplicate, reducing the system from n² to n*(n+1)/2 unknowns.

    # Computes kron(X, X) and vec(B)
    vec_B = B.flatten()
    kron_X = cp.kron(X, X)

    # Computes indices to delete
    to_del = []
    if not to_del_path:
        # Stores the dimension of X in matrix_dim
        matrix_dim = cp.shape(X)[0]

        processed_cols = []
        for col in range(matrix_dim ** 2):
            # Map the flat column index back to (row, col) coordinates in X
            col_idx = col // matrix_dim
            row_idx = col % matrix_dim

            # If the row and column are not the same, then we mark the duplicate for deletion.
            if col_idx != row_idx:
                # The mirror entry swaps row and col coordinates in the flat index
                mirror_col = int(row_idx * matrix_dim + col_idx)

                # If the column has not been processed...
                if mirror_col not in processed_cols:
                    # adds it to the list of columns to delete...
                    to_del.append(mirror_col)
                    # and sums it with its partner column.
                    kron_X[:, col] += kron_X[:, mirror_col]
                # Mark the column as processed.
                processed_cols.append(col)
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
    """Applies the inverse function of symmetric modification to vec(A)."""

    A = cp.zeros((orig_dim, orig_dim))

    upper_tri_flat_indices = []
    for i in range(orig_dim):
        # Creates list of indices in upper right triangular matrix
        row_indices = [(i * orig_dim + j + i) for j in range(orig_dim - i)]
        upper_tri_flat_indices += row_indices

    upper_tri_2d_indices = []
    for flat_idx in upper_tri_flat_indices:
        col = flat_idx // orig_dim
        row = flat_idx % orig_dim
        upper_tri_2d_indices.append([col, row])

    for (flat_pos, matrix_coord) in enumerate(upper_tri_2d_indices):
        A[matrix_coord[0], matrix_coord[1]] = vec_A[flat_pos]

    A_complement = cp.copy(A.T)
    cp.fill_diagonal(A_complement, 0)
    A += A_complement
    return A
