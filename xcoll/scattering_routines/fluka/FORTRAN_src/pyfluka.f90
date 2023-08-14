subroutine pyfluka_init(n_alloc)
    use mod_fluka
    !, only : fluka_enable, fluka_mod_init
    use physical_constants, only : clight

    implicit none
    integer, intent(in)    :: n_alloc

    call fluka_mod_init(n_alloc, 500, clight)
    fluka_enable = .true.

end subroutine


subroutine pyfluka_connect()
    use mod_fluka
    !, only : fluka_connect, fluka_connected

    implicit none

    integer fluka_con

    fluka_con = fluka_connect()
    if(fluka_con == -1) then
      PRINT *, "ERROR Cannot connect to Fluka server"
    endif
    PRINT *, "Successfully connected to Fluka server"
    fluka_connected = .true.

end subroutine


subroutine pyfluka_close()
    use mod_fluka
    !, only : fluka_close

    implicit none

    call fluka_close

end subroutine
      

subroutine pyfluka_set_n_alloc(n_alloc)
    use mod_fluka
    !, only : fluka_init_max_uid, fluka_enable

    implicit none
    integer, intent(in)    :: n_alloc
    integer fluka_con

    if(fluka_enable) then
        PRINT *, "Changing n_alloc in FLUKA"
        fluka_con = fluka_init_max_uid( n_alloc )
        if(fluka_con < 0) then
            PRINT *, "Failed to change n_alloc in FLUKA"
        end if
        PRINT *, "Succesfully changed n_alloc in FLUKA"
    end if
end subroutine


subroutine track_fluka(turn, fluka_id, length, alive_part, max_part, x_part, xp_part, y_part, yp_part, &
                       zeta_part, e_part, m_part, q_part, A_part, Z_part, pdgid_part, part_id, parent_id, &
                       part_weight, spin_x_part, spin_y_part, spin_z_part)

    use floatPrecision
    use numerical_constants, only : zero, one, c1e3, c1m3
    use crcoall
    use parpro
    use mod_common
    use mod_common_track
    use mod_common_main
    use mod_fluka

    implicit none

    integer(kind=int32), intent(in)    :: turn
    integer(kind=int32), intent(in)    :: fluka_id
    real(kind=8),        intent(in)    :: length
    integer,             intent(in)    :: alive_part           ! napx
    integer,             intent(in)    :: max_part             ! npart
    real(kind=8),        intent(inout) :: x_part(max_part)     ! [mm]    xv1
    real(kind=8),        intent(inout) :: xp_part(max_part)    ! [1e-3]  yv1
    real(kind=8),        intent(inout) :: y_part(max_part)     ! [mm]    xv2
    real(kind=8),        intent(inout) :: yp_part(max_part)    ! [1e-3]  yv2
    real(kind=8),        intent(inout) :: zeta_part(max_part)  ! [mm]    sigmv
    real(kind=8),        intent(inout) :: e_part(max_part)     ! [MeV]   ejv   (ejfv is momentum, dpsv is delta, oidpsv is 1/(1+d))
    real(kind=8),        intent(inout) :: m_part(max_part)     ! [MeV]   nucm
    integer(kind=int16), intent(inout) :: q_part(max_part)     !         nqq     Charge
    integer(kind=int16), intent(inout) :: A_part(max_part)     !         naa     Ion atomic mass
    integer(kind=int16), intent(inout) :: Z_part(max_part)     !         nzz     Ion atomic number
    integer(kind=int32), intent(inout) :: pdgid_part(max_part) !         pdgid   Particle PDGid
    integer(kind=int32), intent(inout) :: part_id(max_part)
    integer(kind=int32), intent(inout) :: parent_id(max_part)
    real(kind=8),        intent(inout) :: part_weight(max_part)
    real(kind=8),        intent(inout) :: spin_x_part(max_part)  ! spin_x  ! x component of the particle spin
    real(kind=8),        intent(inout) :: spin_y_part(max_part)  ! spin_y  ! y component of the particle spin
    real(kind=8),        intent(inout) :: spin_z_part(max_part)  ! spin_z  ! z component of the particle spin

    integer ret

    npart = max_part
    napx = alive_part

    ret = fluka_send_receive(turn, fluka_id, length, alive_part, max_part, x_part, y_part, xp_part, yp_part, &
                           zeta_part, e_part, A_part, Z_part, m_part, q_part, pdgid_part, &
                           part_id, parent_id, part_weight, spin_x_part, spin_y_part, spin_z_part )
    napx = alive_part

    if (ret.lt.0) then
        PRINT *, 'FLUKA> ERROR ', ret, ' in Fluka communication returned by fluka_send_receive...'
        PRINT *, 'ENDED WITH ERROR.'
    end if

    return
end subroutine track_fluka
