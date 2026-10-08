# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

# Plot the loss maps of B1 and B2, each one alone and both overlaid on the same plot.
#
# The B2 loss map must have been made on the reversed line (line_is_reversed=True), so
# that both beams share the B1 s coordinate.
#
# Example:
#   python lossmap_two_beams.py \
#       --b1 'results/b1/jsons/lossmap_b1_ip1_run_*.json' \
#       --b2 'results/b2/jsons/lossmap_b2_ip1_run_*.json' \
#       --metadata lhc_collimation_metadata.json --zoom IP1 --out plots/lossmaps

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

import xcoll as xc


parser = argparse.ArgumentParser(description="Plot the B1 and B2 loss maps, alone and overlaid.")
parser.add_argument('--b1', required=True, help="B1 loss map JSON file, or a glob of them (merged)")
parser.add_argument('--b2', required=True, help="B2 loss map JSON file, or a glob of them (merged)")
parser.add_argument('--metadata', default=None,
                    help="JSON with cold_regions/warm_regions/s_range, applied to both loss maps")
parser.add_argument('--zoom', nargs='*', default=['IP1'],
                    help="s_range names to zoom on, besides the full ring (default: IP1)")
parser.add_argument('--out', default='plots/lossmaps', help="Output directory")
parser.add_argument('--name', default='lossmap', help="Prefix of the output files")
parser.add_argument('--fix-b2-aperture', action='store_true',
                    help="Mirror the B2 aperture positions (s -> L - s), for loss maps made "
                       + "before the fix of the aperture positions on reversed lines")
args = parser.parse_args()


def load(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No loss map found for '{pattern}'.")
    lm = xc.LossMap.from_json(files, verbose=False)
    print(f"Loaded {len(files)} file(s): {lm}")
    return lm


lm_b1 = load(args.b1)
lm_b2 = load(args.b2)
if not lm_b2.line_is_reversed:
    raise ValueError("The B2 loss map was not made on the reversed line (line_is_reversed=False).")

if args.metadata is not None:
    with open(args.metadata) as fid:
        metadata = json.load(fid)
    lm_b1.update_metadata(metadata)
    lm_b2.update_metadata(metadata)

if args.fix_b2_aperture and not lm_b2.interpolation:
    L = lm_b2.machine_length
    aper_s = np.mod(L - lm_b2._aper_s, L)
    order = np.argsort(aper_s)
    lm_b2._aper_s = aper_s[order]
    for attr in ['_aper_nabs', '_aper_eabs', '_aper_name', '_aper_type', '_aper_length']:
        setattr(lm_b2, attr, getattr(lm_b2, attr)[order])

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for label, lm, kwargs in [('b1',    lm_b1, {}),
                          ('b2',    lm_b2, {}),
                          ('b1_b2', lm_b1, {'b2': lm_b2})]:
    fig, _ = lm.plot(zoom=args.zoom, show=False, **kwargs)
    fig.savefig(out / f'{args.name}_{label}.pdf', bbox_inches='tight')
    print(f"Saved {out / f'{args.name}_{label}.pdf'}")
