import gc
import cupy as cp
import inout
from inout import get_schaefer100_sc, get_schaefer100_fc
from lin_gen_model import Subject, binary_search_train
from transformations import symmetric_modification


def train_subject(subject_id):
    """
    Train the subject-level model for a given subject ID.
    """
    # Load the data
    sc, fc = get_schaefer100_sc(subject_id), get_schaefer100_fc(subject_id)
    sc = cp.log10(sc + 1)
    subject = Subject(subject_id, sc, fc, symmetric_modification)
    rules, alpha = binary_search_train(subject.transformed_sc, subject.transformed_fc, max_iter=10)

    # Save the model
    sl_dir = "subject_level_log10"
    inout.check_paths(sl_dir)
    cp.savetxt(f"{sl_dir}/rules_sub-{subject_id}", rules)

    print(f"Subject {subject_id} finished training with alpha: {alpha}")
    print()

for subject_id in range(1, 51):
    train_subject(subject_id)

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    # Force Python garbage collection:
    gc.collect()
