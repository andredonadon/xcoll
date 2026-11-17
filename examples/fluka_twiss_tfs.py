# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""
MAD-X style thick-lens Twiss file (TFS) of an xsuite line and the collimator settings of a CollDB, as inputs for
the FLUKA LineBuilder. The line must be thick. Without arguments, it uses the HL-LHC v1.9 B1 thick example machine
(round optics, beta* 15 cm) and its CollDB.

    python fluka_twiss_tfs.py [line.json] [twiss.tfs] --colldb CollDB.yaml --beam 1 --sequence LHCB1 \
        --collimator-columns --collgaps collgaps_b1.dat
"""

import argparse
from pathlib import Path

import xtrack as xt
from xcoll.scattering_routines.fluka.twiss_tfs import (S_REF, twiss_table, write_tfs, compare_tfs,
            collimator_settings, write_collgaps, read_collgaps, add_collimator_columns)


path_in = Path(__file__).parent

parser = argparse.ArgumentParser(description=__doc__.strip().split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('line', nargs='?', default=path_in / 'machines' / 'hllhc19_b1_thick.json', help="xsuite line (JSON)")
parser.add_argument('output', nargs='?', default='twiss_b1.tfs', help="TFS file to write")
parser.add_argument('--s-ref', choices=list(S_REF), default='exit',
                    help="where S and the optics are given: exit (default; LineBuilder ref -1) or centre (ref 0)")
parser.add_argument('--colldb', default=path_in / 'colldbs' / 'hllhc_tight_15cm.yaml',
                    help="CollDB: its collimators get KEYWORD COLLIMATOR, its emittances give EX, EY")
parser.add_argument('--beam', type=int, default=1, help="beam of the CollDB (default 1)")
parser.add_argument('--sequence', default='LHCB1', help="@ SEQUENCE")
parser.add_argument('--particle', help="@ PARTICLE (default: from the reference particle of the line)")
parser.add_argument('--nemitt-x', type=float, help="normalised emittance for EX (default: from the CollDB)")
parser.add_argument('--nemitt-y', type=float, help="normalised emittance for EY (default: from the CollDB)")
parser.add_argument('--sige', type=float, default=0., help="@ SIGE")
parser.add_argument('--sigt', type=float, default=0., help="@ SIGT")
parser.add_argument('--npart', type=float, default=0., help="@ NPART")
parser.add_argument('--digits', type=int, default=10, help="significant digits of the numbers (default 10)")
parser.add_argument('--collimator-columns', action='store_true',
                    help="put the collimator settings of --colldb in the Twiss file (ID, HALFGAP, NSIG, TILT1, TILT2, "
                         "LENGTH, and ANGLE = jaw rotation on the collimator rows)")
parser.add_argument('--collimators-from', metavar='COLLGAPS',
                    help="put the settings of this collgaps file in the Twiss file, as --collimator-columns")
parser.add_argument('--collgaps', help="also write the collimator settings of --colldb in the collgaps format "
                                       "(LineBuilder inputcard: collimators_settings; FLUKA_builder: --CollG)")
parser.add_argument('--compare', help="TFS file to compare the output with")
parser.add_argument('--compare-only', action='store_true', help="only compare the existing output with --compare")
args = parser.parse_args()


if not args.compare_only:
    line = xt.load(args.line)
    if isinstance(line, xt.Environment):
        line = line[args.sequence.lower()]
    table = twiss_table(line, s_ref=args.s_ref, colldb=args.colldb, beam=args.beam, sequence=args.sequence,
                        particle=args.particle, nemitt_x=args.nemitt_x, nemitt_y=args.nemitt_y,
                        sige=args.sige, sigt=args.sigt, npart=args.npart)
    settings = None
    if args.collimator_columns or args.collgaps:
        settings = collimator_settings(table, args.colldb, beam=args.beam, s_ref=args.s_ref,
                                       nemitt_x=args.nemitt_x, nemitt_y=args.nemitt_y)
    if args.collgaps:
        write_collgaps(args.collgaps, settings)
        print(f"Written {args.collgaps}: settings of {len(settings)} collimators.")
    if args.collimators_from:
        add_collimator_columns(table, read_collgaps(args.collimators_from))
    elif args.collimator_columns:
        add_collimator_columns(table, settings)
    write_tfs(args.output, table, digits=args.digits)
    print(f"Written {args.output}: {len(table['name'])} rows, S at the element {args.s_ref} "
          f"(LineBuilder twiss_file ref {S_REF[args.s_ref]}).")
if args.compare:
    compare_tfs(args.output, args.compare)
