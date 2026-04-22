"""
Brain region definitions for the Schaefer100 + subcortical atlas (114-node space).
"""

REGION_LABELS = [
    'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Accumbens', 'Amygdala',
    'Hypocampus', 'Sommatosensory', 'Visual Cortex', 'DAN', 'SAN', 'Limbic',
    'Cont', 'DMN'
]

BRAIN_REGIONS = {
    'Thalamus':       (0, 7),
    'Caudate':        (1, 8),
    'Putamen':        (2, 9),
    'Pallidum':       (3, 10),
    'Accumbens':      (6, 13),
    'Amygdala':       (5, 12),
    'Hypocampus':     (4, 11),
    'Sommatosensory': list(range(24, 29)) + list(range(73, 80)),
    'Visual Cortex':  list(range(15, 23)) + list(range(65, 72)),
    'DAN':            list(range(30, 37)) + list(range(81, 87)),
    'SAN':            list(range(38, 44)) + list(range(88, 92)),
    'Limbic':         list(range(45, 47)) + list(range(93, 94)),
    'Cont':           list(range(48, 51)) + list(range(95, 103)),
    'DMN':            list(range(52, 64)) + list(range(104, 114)),
}
