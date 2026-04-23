import cupy as cp
import gc
import inout
from subject import Subject
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

    rules = cp.linalg.pinv(X) @ y

    # Save the model
    sl_dir = "group_level_log10"
    inout.check_paths(sl_dir)
    cp.savetxt(f"{sl_dir}/rules", rules)


if __name__ == "__main__":
    subject_ids = [subject_id for subject_id in range(1, 51)]
    train_group(subject_ids)
