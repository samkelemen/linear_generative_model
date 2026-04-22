"""
Project-wide constants for the linear generative model.
"""

# Schaefer100 atlas with medial wall removed: 116 - 2 = 114 nodes
N_NODES = 114

# Number of healthy control subjects in the dataset
N_SUBJECTS = 50

# Medial wall node indices to remove from the Schaefer100 atlas (0-indexed, pre-removal)
MEDIAL_WALL_INDICES = [14, 65]

# Subject IDs for the tumor dataset
TUMOR_SUBJECT_IDS = [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17,
                     19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 31]
