// copyright ############################### #
// This file is part of the Xcoll Package.   #
// Copyright (c) CERN, 2024.                 #
// ######################################### #

#ifndef XCOLL_BLOWUP_H
#define XCOLL_BLOWUP_H

/*gpufun*/
void BlowUp_track_local_particle(BlowUpData el, LocalParticle* part0){

    int8_t plane     = BlowUpData_get__plane(el);
    double kick_rms  = BlowUpData_get__kick_rms(el);
    int8_t active    = BlowUpData_get__active(el);

    //start_per_particle_block (part0->part)
        if (active){
            double kick = kick_rms * (2*RandomUniform_generate(part) - 1);
            if (plane == 1){
                LocalParticle_add_to_px(part, kick);
            } else if (plane == -1){
                LocalParticle_add_to_py(part, kick);
            } else {
                LocalParticle_kill_particle(part, XC_ERR_INVALID_XOFIELD);
            }
        }
    //end_per_particle_block
}

#endif /* XCOLL_BLOWUP_H */
