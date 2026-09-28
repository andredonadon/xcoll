import time
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import xtrack as xt
import xcoll as xc
import xpart as xp


path_in = Path(__file__).parent
path_out = Path.cwd()

beam = 1
ions = True
dump = False


def other_beam_at_ip(tw, ip, line_b2=None):
    # The FLUKA source expects the optics and closed orbit of the other beam in its
    # own reference frame (s along its motion, x mirrored). Passing the tracked
    # beam's twiss instead mirrors its crossing angle and removes the crossing.
    if line_b2 is not None:
        # Counter-rotating line, in its own frame
        row = line_b2.twiss().rows[ip]
        return {kk: row[kk][0] for kk in ['x', 'px', 'y', 'py', 'betx', 'bety',
                                          'alfx', 'alfy', 'dx', 'dpx', 'dy', 'dpy']}
    # Fallback: mirror the tracked beam through the IP, assuming the two beams are
    # mirror images of each other (LHC design; the crossing angles of the real b1/b2
    # lines agree within 0.3% at IP1/2/5/8). In the lab the other beam's orbit is
    # the transverse reflection of this one; going to its own frame (s -> -s, x -> -x)
    # gives x2 = x1, px2 = -px1, y2 = -y1, py2 = py1 (dispersion transforms like the
    # orbit). Beta and alpha are copied (alpha only enters FLUKA as alpha**2).
    row = tw.rows[ip]
    return {'x':     row.x[0],   'px':   -row.px[0],
            'y':    -row.y[0],   'py':    row.py[0],
            'betx':  row.betx[0], 'bety': row.bety[0],
            'alfx':  row.alfx[0], 'alfy': row.alfy[0],
            'dx':    row.dx[0],  'dpx':  -row.dpx[0],
            'dy':   -row.dy[0],  'dpy':   row.dpy[0]}


def FlukaBeamBeamSource(line, colldb, int_type, ip, num_particles, pdg_id_b1, p0c_b1, Z_b2, A_b2,
                        sigma_p_x2 = 0, sigma_p_y2 = 0, sigma_z = 0, sigma_dpp = 0, line_b2 = None):

    _capacity = num_particles * 10000
    tw = line.twiss()
    b2 = other_beam_at_ip(tw, ip, line_b2=line_b2)
    if ip == "ip1":
        xsx = line["on_x1"]
        xsv = 0
        phi_ir = line['phi_ir1']
    elif ip == "ip2":
        xsx = line["on_x2h"]
        phi_ir = line['phi_ir2']
        xsv = line["on_x2v"]
    elif ip == "ip5":
        xsx = line["on_x5"]
        xsv = 0
        phi_ir = line['phi_ir5']
    elif ip == "ip8":
        xsx = line["on_x8h"]
        xsv = line["on_x8v"]
        phi_ir = line['phi_ir8']

    theta2 = (math.sqrt(xsx**2 + xsv**2) * 1e-6)

    # if theta2 is negative, make it positive and change sing to phi_ir
    if xsx < 0 or xsv < 0:
        theta2 = -theta2
        phi_ir = -phi_ir

    geo_emittance_x = tw.beta0*tw.gamma0*colldb.nemitt_x
    geo_emittance_y = tw.beta0*tw.gamma0*colldb.nemitt_y

    sigma_p_x2 = (geo_emittance_x*(tw.rows[ip].gamx)[0])**0.5
    sigma_p_y2 = (geo_emittance_y*(tw.rows[ip].gamy)[0])**0.5

    # FLUKA samples the collision position with the luminous-region length, not the
    # bunch length: for two Gaussian bunches of rms length sigma_z crossing with
    # half-angles theta_x/y (Piwinski, hourglass neglected)
    #   sigma_lum = sigma_z / sqrt(2) / sqrt(1 + (theta_x sigma_z/sigma_x)^2 + (theta_y sigma_z/sigma_y)^2)
    # The half-angles follow from both beams' IP angles (the other beam in its own frame).
    # Geometric emittance with the tracked species' beta*gamma (the line's own reference
    # particle may be a proton; the optics are the same at equal rigidity).
    ref_b1 = xt.Particles.reference_from_pdg_id(pdg_id=pdg_id_b1, p0c=p0c_b1)
    betagamma_b1 = ref_b1.beta0[0] * ref_b1.gamma0[0]
    emit_geo_x = colldb.nemitt_x / betagamma_b1
    emit_geo_y = colldb.nemitt_y / betagamma_b1
    sigma_x_ip = (tw.rows[ip].betx[0] * emit_geo_x)**0.5
    sigma_y_ip = (tw.rows[ip].bety[0] * emit_geo_y)**0.5
    theta_x = 0.5 * (tw.rows[ip].px[0] - b2['px'])
    theta_y = 0.5 * (tw.rows[ip].py[0] + b2['py'])
    piwinski_sq = (theta_x * sigma_z / sigma_x_ip)**2 + (theta_y * sigma_z / sigma_y_ip)**2
    sigma_z_lum = sigma_z / math.sqrt(2) / math.sqrt(1 + piwinski_sq)
    print(f"Luminous region: sigma_z_lum = {sigma_z_lum*1e2:.2f} cm (bunch sigma_z = {sigma_z*1e2:.2f} cm, "
          f"half crossing = ({theta_x*1e6:.1f}, {theta_y*1e6:.1f}) urad, Piwinski = {math.sqrt(piwinski_sq):.3f})")


    bb_int = {
            'int_type': int_type,
            'theta2': theta2, # Angle between b2 and -Z
            'xs': phi_ir,
            'sigma_p_x2': sigma_p_x2,
            'sigma_p_y2': sigma_p_y2,
            'Z': Z_b2,
            'A': A_b2,
            'betx': b2['betx'],
            'bety': b2['bety'],
            'dx': b2['dx'],
            'dy': b2['dy'],
            'alfx': b2['alfx'],
            'alfy': b2['alfy'],
            'dpx': b2['dpx'],
            'dpy': b2['dpy'],
            'rms_emx': colldb.nemitt_x, #/(tw.beta0* tw.gamma0),
            'rms_emy': colldb.nemitt_y, #/(tw.beta0* tw.gamma0)
            'offset': [b2['x'], b2['px'], b2['y'], b2['py']],
            'sigma_z': sigma_z_lum * 1e2,  # collision-position spread [cm]
            'sigma_dpp': sigma_dpp
            }

    coll_dummy = xc.FlukaCollimator(assembly="lhc_ippipe", length=0.6, jaw=0.001)

    xc.fluka.engine.particle_ref = xt.Particles.reference_from_pdg_id(pdg_id=pdg_id_b1, p0c=p0c_b1)
    xc.fluka.engine.minimum_free_length_fortran_array = _capacity
    xc.fluka.engine.relative_length_fortran_array = 200 if ions else 20

    # xc.fluka.engine.seed = 5656565
    # FLUKA generates the collision in the rest frame of the other beam's nucleus, i.e.
    # at ~2 gamma^2 m per nucleon (~1.5e7 GeV/n for Pb-Pb at 6.8 Z TeV). By default the
    # high-energy model (DPMJET, used for ion-ion) is only initialised up to the BEAM-card
    # (fixed-target) energy and crashes. PHYSICS/LIMITS sets the maximum pp CMS momentum,
    # which for colliding beams is the beam momentum per nucleon (+ margin).
    part_ref = xc.fluka.engine.particle_ref
    A_b1 = max(1, round(part_ref.mass0 / 931.49410242e6))         # mass / atomic mass unit
    p_n_b1 = p0c_b1 / A_b1 / 1e9                                   # [GeV/c per nucleon]
    p_n_b2 = p0c_b1 / part_ref.q0 * Z_b2 / A_b2 / 1e9              # same rigidity
    p_cms_max = 1.1 * max(p_n_b1, p_n_b2)
    limits_card = f"PHYSICS   {p_cms_max:10.1f}{'':50s}LIMITS"

    xc.fluka.engine.start(elements=coll_dummy, clean=False , verbose=True, include_showers=False, return_ions=True, bb_int=bb_int, touches=False,
                          extra_physics_cards=[limits_card])


    line.build_tracker()

    nemitt_x = colldb.nemitt_x
    nemitt_y = colldb.nemitt_y
    bunch_intensity = 1.8E11
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=num_particles, total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
        line=line, particle_ref=xc.fluka.engine.particle_ref)
    tt = line.get_table()
    part_init = line.build_particles(x=particles.x, y=particles.y,
                            px=particles.px,
                            py=particles.py,
                            zeta=particles.zeta,
                            delta=particles.delta,
                            nemitt_x=colldb.nemitt_x,
                            nemitt_y=colldb.nemitt_y,
                            at_element=ip,
                            match_at_s=tt['s', ip],
                            particle_ref=xc.fluka.engine.particle_ref,
                            _capacity=_capacity,
    mode="normalized_transverse")
    part = part_init.copy()

    print(f"Tracking {num_particles} particles (FLUKA)...     ", end='')
    start = time.time()
    coll_dummy.track(part)
    print(f"Done in {round(time.time()-start, 3)}s.")
    xc.fluka.engine.stop(clean=False)

    line.discard_tracker()

    return part


start_time = time.time()

# Path to inputs
path_out.mkdir(exist_ok=True)

num_particles = int(100)

line = xt.load(path_in / 'machines' / f'levelling.23_b{beam}.json')
# Counter-rotating beam (in its own frame), for the optics/orbit of the other beam at the IP.
# Set to None to mirror the tracked beam instead (assumes the beams are mirror images).
line_b2 = xt.load(path_in / 'machines' / f'levelling.23_b{3 - beam}.json')
colldb = xc.CollimatorDatabase.from_yaml(path_in / "colldbs" / f'levelling.23.yaml', beam=beam)

if not ions:
    # p-p example
    part = FlukaBeamBeamSource(
        line=line,
        line_b2=line_b2,
        colldb=colldb,
        int_type=1.0, # 1.0 inelastic, 10.0 elastic, 100.0 Emd
        ip="ip1",
        num_particles=num_particles,
        pdg_id_b1='proton',
        p0c_b1=6.8e12,
        Z_b2=1,
        A_b2=1,
        sigma_z = 8e-2,
        sigma_dpp = 1.1e-4
    )

if ions:
    # Pb-Pb example
    part = FlukaBeamBeamSource(
        line=line,
        line_b2=line_b2,
        colldb=colldb,
        int_type=1.0, # 1.0 inelastic, 10.0 elastic, 100.0 Emd
        ip="ip1",
        num_particles=num_particles,
        pdg_id_b1='Pb208',
        p0c_b1=6.8e12*82,
        Z_b2=82,
        A_b2=208,
        sigma_z = 8e-2,
        sigma_dpp = 1.1e-4
    )


part_alive = part.filter(part.state == 1)
values, counts = np.unique(part_alive.pdg_id, return_counts=True)

# Bar chart with string labels
labels = [str(v) for v in values]

plt.figure(figsize=(6,4))
plt.yscale('log')
plt.bar(labels, counts, color='skyblue', edgecolor='black')
plt.xlabel('Particles')
plt.ylabel('Counts')
plt.tight_layout()
plt.xticks(rotation=90)
        
plt.show()

if dump:
    import pickle
    part_pkl_init = path_out / f'part_init.pkl'
    part_dpp_cut_init = part.filter(part.pdg_id != -999999999)
    with open(part_pkl_init, 'wb') as fid_out:
        pickle.dump(part_dpp_cut_init.to_dict(), fid_out)

