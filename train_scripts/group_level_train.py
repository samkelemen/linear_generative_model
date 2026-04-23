import numpy as np
import cupy as cp
import gc
import inout
from sklearn.linear_model import LinearRegression
from lin_gen_model import Subject
from transformations import symmetric_modification
from inout import get_schaefer100_sc, get_schaefer100_fc


def train_group(subject_ids):

    transformed_scs = []
    transformed_fcs = []

    for subject_id in subject_ids:
        sc, fc = get_schaefer100_sc(subject_id), get_schaefer100_fc(subject_id)
        sc = cp.log10(sc + 1)
        subject = Subject(subject_id, sc, fc, symmetric_modification)

        transformed_scs.append(subject.transformed_sc.copy())
        transformed_fcs.append(subject.transformed_fc.copy())

        del sc, fc, subject
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        # Force Python garbage collection:
        gc.collect()

    X = cp.vstack(transformed_scs).T
    y = cp.hstack(transformed_fcs)

    del transformed_scs, transformed_fcs
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    # Force Python garbage collection:
    gc.collect()

    a = cp.linalg.pinv(transformed_scs[:6555, :6555*30])

    #rules = LinearRegression(fit_intercept=False).fit(transformed_scs.T.astype(np.float32), transformed_fcs.astype(np.float32))

    # Save the model
    #sl_dir = "group_level_log10"
    #inout.check_paths(sl_dir)
    #np.savetxt(f"{sl_dir}/rules", rules)

subject_ids = [subject_id for subject_id in range(1, 51)]
train_group(subject_ids)
