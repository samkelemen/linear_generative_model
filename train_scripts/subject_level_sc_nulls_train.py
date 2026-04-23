import gc
import cupy as cp
import inout
from inout import get_schaefer100_fc
from lin_gen_model import binary_search_train
from subject import Subject
from transformations import symmetric_modification


def train_subject(subject_id):
    """
    Train the subject-level model for a given subject ID.
    """
    fc = get_schaefer100_fc(subject_id)

    for null_num in range(0, 100):

        # Load the data
        sc = cp.loadtxt(f"sc_nulls/{subject_id}/null_X{null_num}")
        sc = inout.remove_medial_wall(sc)
        subject = Subject(subject_id, sc, fc, symmetric_modification)
        rules, alpha = binary_search_train(
            subject.transformed_sc, subject.transformed_fc, max_iter=10
        )

        # Save the model
        sl_dir = f"rule_nulls/{subject_id}"
        inout.check_paths(sl_dir)
        cp.savetxt(f"{sl_dir}/null_rules{null_num}", rules)

        print(
            f"Subject {subject_id}, null_num {null_num} finished training with alpha: {alpha}"
        )
        print()
        gc.collect()


if __name__ == "__main__":
    for subject_id in range(1, 51):
        train_subject(subject_id)

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        # Force Python garbage collection:
        gc.collect()
