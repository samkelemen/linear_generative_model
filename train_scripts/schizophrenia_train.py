import cupy as cp
from lin_gen_model import binary_search_train
from subject import Subject
from transformations import symmetric_modification
from train_functions import get_mat
import inout

sc_ctrl_path = "SC_FC_Connectomes/SC_number_of_fibers/ctrl"
sc_schz_path = "SC_FC_Connectomes/SC_number_of_fibers/schz"
fc_ctrl_path = "SC_FC_Connectomes/FC_correlation/ctrl"
fc_schz_path = "SC_FC_Connectomes/FC_correlation/schz"
sc_dens_ctrl_path = "SC_FC_Connectomes/SC_density/ctrl"
sc_dens_schz_path = "SC_FC_Connectomes/SC_density/schz"

if __name__ == "__main__":
    GROUPS = ("schz", "cntrl")

    for group in GROUPS:
        for subject_id in range(1, 28):
            # Load the data for the subject
            if group == "schz":
                sc = get_mat(sc_schz_path, subject_id)
                fc = get_mat(fc_schz_path, subject_id)
            else:
                sc = get_mat(sc_ctrl_path, subject_id)
                fc = get_mat(fc_ctrl_path, subject_id)
            cp.fill_diagonal(fc, 1)

            # Create the subject object.
            subject = Subject(subject_id, sc, fc, symmetric_modification)

            # Make sure the output directory exists. If not created it.
            outdir = f"schz_results/{group}/"
            inout.check_paths(outdir)

            # Train
            rules, alpha = binary_search_train(
                subject.transformed_sc, subject.transformed_fc
            )

            print(f"Subject {subject_id} finished training with alpha: {alpha}")
            print()

            cp.savetxt(outdir + f"rules_sub-{subject_id}", rules)
