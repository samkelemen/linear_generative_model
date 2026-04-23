import numpy as np
import os
from train_functions import compute_pvals, compute_sig_rules


for subject_id in range(1, 51):
    rules = np.loadtxt(f"subject_level/rules_sub-{subject_id}")

    distribution = []
    for null_num in range(100):
        if not os.path.exists(f"rule_nulls/{subject_id}/null_rules{null_num}"):
            continue
        null_rules = np.loadtxt(f"rule_nulls/{subject_id}/null_rules{null_num}")
        distribution.append(null_rules)
    distribution = np.vstack(distribution).T

    pvals = compute_pvals(distribution, rules)
    sig_rules = compute_sig_rules(pvals, rules)

    np.savetxt(f"p_values_sub-{subject_id}", pvals)
    np.savetxt(f"sig_rules_sub-{subject_id}", sig_rules)
