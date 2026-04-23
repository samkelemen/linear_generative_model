import gc
import cupy as cp

from lin_gen_model import calc_alpha_grid, lasso_regression
from subject import Subject
from transformations import symmetric_modification
from train_functions import load_connectomes
import inout


def validation_train(subject_id):
    """
    Train the subject-level model for a given subject ID at different window sizes and penalty values.
    """
    time_steps = 587
    sc, fc = load_connectomes(subject_id, time_steps=time_steps)
    subject = Subject(subject_id, sc, fc, symmetric_modification)
    alpha_grid = calc_alpha_grid(subject.transformed_sc, subject.transformed_fc, 20)

    print(f"Window = {time_steps}, alpha_grid = {alpha_grid}")

    for num, alpha in enumerate(alpha_grid):
        rules = lasso_regression(
            subject.transformed_sc, subject.transformed_fc, alpha, max_iter=100
        )
        outdir = f"sc_density_sweeps/{time_steps}/"
        inout.check_paths(outdir)
        cp.savetxt(f"{outdir}rules_sub-{subject_id}_lambda-{num}", rules)

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        # Force Python garbage collection:
        gc.collect()


SUBJECT_IDs = ["032301", "032304", "032307"]
for subject_id in SUBJECT_IDs:
    validation_train(subject_id)
