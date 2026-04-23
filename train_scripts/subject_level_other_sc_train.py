import gc
import cupy as cp
import inout
from inout import get_schaefer100_sc, get_schaefer100_fc
from lin_gen_model import binary_search_train
from subject import Subject
from transformations import symmetric_modification


def train_subject(subject_id):
    """
    Train the subject-level model for a given subject ID.
    """
    fc = get_schaefer100_fc(subject_id)

    # Map each subject_id to another in a one-to-one function
    if subject_id <= 40:
        subject_id_sc = subject_id + 10
    else:
        subject_id_sc = (subject_id - 1) % 40 + 1

    # Load the data
    sc = get_schaefer100_sc(subject_id_sc)
    subject = Subject(subject_id, sc, fc, symmetric_modification)
    rules, alpha = binary_search_train(
        subject.transformed_sc, subject.transformed_fc, max_iter=10
    )

    # Save the model
    sl_dir = "subject_level_other_sc"
    inout.check_paths(sl_dir)
    cp.savetxt(f"{sl_dir}/rules_sc-{subject_id_sc}_fc-{subject_id}", rules)

    print(
        f"Subject {subject_id} FC w/ Subject {subject_id_sc} SC finished training with alpha: {alpha}"
    )
    print()


for subject_id in range(1, 51):
    train_subject(subject_id)

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    # Force Python garbage collection:
    gc.collect()
