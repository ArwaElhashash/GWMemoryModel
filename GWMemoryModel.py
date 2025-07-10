#!/usr/bin/env python


# This code implements the waveform model developed in the following two papers:
# Paper I: Arwa Elhashash and David A. Nichols. ``Waveform models for the gravitational-wave memory effect: Extreme mass-ratio limit and final memory offset''. Phys. Rev. D 111, 044052 (2025). ArXiv:2407.19017.
# Paper II: Arwa Elhashash and David A. Nichols. ``Waveform models for the gravitational-wave memory effect: II. Time-domain and frequency-domain models for nonspinning binaries''. ArXiv:2504.18635.

# Needed packages
import numpy as np
import qnm
import surfinBH
from typing import Tuple

# Loading the remnant fit for final mass and spin calculation
remnant_fit = surfinBH.LoadFits('NRSur3dq8Remnant')


# # Time-domain Waveform Model

# ## Memory Model Parameters

# Coefficients for the inspiral memory model (Table III. in paper II)
x_insp = np.asarray([2155.81002984, -22124.15922533, 117604.79338105, -331057.67271698, 380724.41433229])

# Coefficients of the QNM fits for the three oscillatory modes with (l, m) = (2, 1), (2, 2) and (3, 2) (Table V. in Paper II)
# C_lmj[(l,m,j)][n] = an array of length N+1 containing the different values of C_lmj (where N is the highest overtone number)

C_lmj = {(2,1,0): np.asarray([ 7.47640117e-03-4.77861746e-02j, -3.05298784e-01+4.81131374e-01j,
         1.71408140e+00-3.41178737e+00j, -4.38123073e+00+1.54558797e+01j,
         5.38391087e+00-3.90554331e+01j, -2.20879082e+00+5.37004343e+01j,
        -1.10505284e+00-3.76743137e+01j,  8.99073412e-01+1.05575520e+01j]),
          (2,1,1): np.asarray([-1.61018486e+00+1.75169623e+00j,  1.26907283e+01-7.75519054e+00j,
        -4.36825771e+01+5.50613602e+01j,  7.91839694e+01-2.65210733e+02j,
        -4.58450711e+01+6.83041494e+02j, -6.25966255e+01-9.42052862e+02j,
         1.03723841e+02+6.59987980e+02j, -4.16504962e+01-1.84599110e+02j]),
          (2,1,2): np.asarray([ 7.24178719e-02-6.65547575e+00j, -2.00629217e+01+5.34017698e+01j,
         5.54302006e+01-2.96790450e+02j,  1.23485122e+02+1.16454473e+03j,
        -8.40503635e+02-2.70119049e+03j,  1.58678621e+03+3.48957089e+03j,
        -1.31238760e+03-2.33163990e+03j,  4.07696371e+02+6.28972040e+02j]),
          (2,2,0): np.asarray([ -0.29753435-1.15857980e-01j,   0.9125314 +1.45773699e+00j,
         -0.12953588-5.91873529e+00j,  -4.28231567+1.52720930e+01j,
         10.7033551 -2.67597483e+01j, -13.51354516+2.95409721e+01j,
          9.02696155-1.84786210e+01j,  -2.41503666+5.00346576e+00j]),
          (2,2,1): np.asarray([8.14452827  +4.447051j  ,  -22.81756451 -34.48048005j,
          33.00384625+123.72646169j,  -31.09575754-290.16484052j,
          37.28279912+453.46760428j,  -38.01471376-441.06585348j,
          17.74458086+243.49435464j,   -2.93832801 -59.43293943j]),
          (2,2,2): np.asarray([-33.15346389-1.23459336e+00j,  135.49649103+9.98933879e+01j,
        -259.62828735-4.33901766e+02j,  386.2277925 +1.01331508e+03j,
        -540.13060856-1.50006992e+03j,  531.01352882+1.35721194e+03j,
        -279.53503036-6.93905934e+02j,   60.7059195 +1.58688807e+02j]),
          (3,2,0): np.asarray([0.03585245+1.06750582e-02j,  -0.10142351-1.91406748e-01j,
         -0.46171425+2.49274286e-01j,   2.90524631+2.08682375e+00j,
         -6.86733489-8.20360778e+00j,   8.70979274+1.25701948e+01j,
         -5.79973071-9.07465431e+00j,   1.581596  +2.55730562e+00j]),
          (3,2,1): np.asarray([-0.62432076  +0.49412758j,    4.42891107  +0.58574827j,
          -1.54492844  +1.4543073j ,  -32.62997708 -50.0505845j ,
          99.37509924+170.79690421j, -139.80675387-254.52251557j,
          99.11121771+182.2816495j ,  -28.26291642 -51.25712928j]),
          (3,2,2): np.asarray([ 2.31826454-2.30899254e+00j,  -16.03196522-1.21651982e+00j,
         -13.31031277-7.07168742e+00j,  218.43441111+1.98088538e+02j,
        -598.91828811-6.66300780e+02j,  811.52998102+9.81025956e+02j,
        -561.09237015-6.94459624e+02j,  156.84119469+1.93151340e+02j])
         }

# Coefficients for the intermediate memory model (Table IV. in paper II)
x_int = np.asarray([8.49306e-1, 1.02024e1, 4.29098e-2, 2.36601e-1, 8.37061e-3, 1.59611e-2, 5.97250e-4, 1.10423e-3, 8.23572e-1, 1.01875e0, 9.40991e-4, 6.22363e-3, 1.19119e-3])


# ## Inspiral Memory

def compute_pn_waveform(tc: float,
                        time: np.ndarray,
                        eta: float)->np.ndarray:
    """
    Compute post-Newtonian waveform

    Args:
        tc: Coalescence time
        time: Time array
        eta: Symmetric mass ratio of the binary system

    Returns:
        Post-Newtonian waveform array
    """
    theta = (eta/5) * (tc - time)
    xpn = (1/4) * theta**(-1/4)

    # Compute PN terms
    pn_terms = {
        'leading': 1,
        '1pn': xpn * (-(4075/4032) + eta*(67/48)),
        '2pn': xpn**2 * (
            -(151877213/67060224) - 
            eta*(123815/44352) + 
            eta**2 * (205/352)
        ),
        '2.5pn': np.pi * xpn**(5/2) * (-(253/336) + eta * (253/84)),
        '3pn': xpn**3 * (
            -(4397711103307/532580106240) + 
            ((700464542023/13948526592) - (205/96) * np.pi**2) * eta +
            (69527951/166053888) * eta**2 +
            (1321981/5930496) * eta**3
        ),
        '3.5pn': np.pi * xpn**(7/2) * (
            (38351671/28740096) - 
            eta*(3486041/598752) - 
            eta**2 * (652889/598752)
        )
    }

    return (4/7) * np.sqrt((5*np.pi)/6) * eta * xpn * sum(pn_terms.values())

def compute_inspiral_memory_model(q: float,
                                  x: np.ndarray = x_insp,
                                  ti: float = -1e4,
                                  tf: float = -2e3,
                                  dt: float = 0.01)-> Tuple[np.ndarray, float]:
    
    '''
    Compute the inspiral memory model
    
    Args:
        q : Mass ratio of the binary system (1 ≤ q ≤ 8)
        x : Array of optimized inspiral parameters
        ti : Start time of inspiral phase
        tf : End time of inspiral phase
        dt: Time step
        
    Returns:
        Tuple of (inspiral model waveform array, coalescence time value)

    '''
    insp_time = np.linspace(ti, tf, 1 + int(np.rint((tf - ti) / dt)))
    
    # Compute symmetric mass ratio
    eta = q / (q + 1)**2
    
    # Compute coalescence time
    powers = np.vstack([eta**i for i in range(len(x))])
    tc = np.dot(x, powers)
    
    # Compute post-Newtonian waveform
    hmem_inspiral = compute_pn_waveform(tc, insp_time, eta)

    return hmem_inspiral, tc


# ## Ringdown Memory

def compute_qnm_parameters(q: float)-> Tuple[float, dict, dict]:
    """
    Compute quasi-normal mode (QNM) parameters for a black-hole binary: mixing coefficients and QNM frequencies.
    
    Args:
        q : Mass ratio of the binary system (1 ≤ q ≤ 8)
        
    Returns:
        Tuple of:
            - mf : Final mass of the remnant black hole
            - A : Mixing coefficients for A_{l'lm}(a\omega_{lmn}) = A[(ll,l,m)] modes
            - omega : QNM frequencies for (l=2, m=1), (l=2, m=2), and (l=3, m=2) modes
    """
    # Setting the spins of both black holes to be zero (non-spinning case)
    chiA, chiB = [0.,0.,0.], [0.,0.,0.]
    
    # Calculate final mass and dimensionless spin of remnant black hole
    mf, mf_err = remnant_fit.mf(q, chiA, chiB) # Final mass
    chif, chif_err = remnant_fit.chif(q, chiA, chiB) # Final spin vector
    # Extract dimensionless spin parameter (z-component)
    a = chif[2] 
    s = -2 # Spin-weight for gravitational wave perturbations
    
    # Extracting mixing coefficients (A_{l'lm}) and QNM frequencies (omega_{lmn})
    A221, A222, A322, A232, A332 = {}, {}, {}, {}, {}
    omega_21, omega_22, omega_32 = {}, {}, {}
    
    # Compute QNM parameters for (l=2, m=1), (l=2, m=2), and (l=3, m=2) modes, looping over the overtone number, n.
    for n in range(0,8):
        omega_21[n] = qnm.modes_cache(s, 2, 1, n)(a)[0]
        A221[n] = qnm.modes_cache(s, 2, 1, n)(a)[2][0]
        
        omega_22[n] = qnm.modes_cache(s, 2, 2, n)(a)[0]
        A222[n], A322[n] = qnm.modes_cache(s, 2, 2, n)(a)[2][0:2]
        
        omega_32[n] = qnm.modes_cache(s, 3, 2, n)(a)[0]
        A232[n], A332[n] = qnm.modes_cache(s, 3, 2, n)(a)[2][0:2]


    # Storing mixing coefficients A_{l'lm}(a\omega_{lmn}) = A[(ll,l,m)]
    A = {
        (2,2,2): A222,
        (3,2,2): A322,
        (2,3,2): A232,
        (3,3,2): A332,
        (2,2,1): A221,
        (3,2,1): [0.]*len(A221),
    }
    
    # Storing QNM frequencies \omega_{lmn} = omega[(l,m)]
    omega = {
        (2,1): omega_21,
        (2,2): omega_22,
        (3,2): omega_32
    }
    
    return mf, A, omega

def compute_ringdown_memory_model(q: float,
                                  C_lmnj: dict = C_lmj,
                                   ti: float = 0.,
                                   tf: float = 1.3e2,
                                   dt: float = 0.01):
    """
    Compute the ringdown memory model
        
    Args:
        q : Mass ratio of the binary system (m1/m2 where m1 >= m2)
        ti : Start time for the ringtown phase
        tf : End time for the ringtown phase 
        dt : Time step
        
    Returns:
        np.ndarray: Ringdown memory waveform model
    """
    # Setting the spins of both black holes to be zero (non-spinning case)
    chiA, chiB = [0.,0.,0.], [0.,0.,0.]
    
    # Compute symmetric mass ratio
    eta = q/(q + 1)**2
    # Final memory strain fit (Table II. of Paper II)
    Delta_hmem_fit = (0.102414 *eta + 0.824195 * eta**2 - 2.3413 * eta**3 
                      + 22.486 * eta**4 - 58.276 * eta**5 + 105.885 * eta**6)
    
    # Compute ringdown qnm parameters
    mf, A, omega = compute_qnm_parameters(q=q)
    
    t = np.linspace(ti, tf, 1 + int(np.rint((tf - ti) / dt)))
    
    C_21 = np.sqrt(1-4*eta) * (C_lmnj[(2,1,0)] + C_lmnj[(2,1,1)] * eta + C_lmnj[(2,1,2)] * eta**2)
    C_22 = C_lmnj[(2,2,0)] + C_lmnj[(2,2,1)] * eta + C_lmnj[(2,2,2)] * eta**2
    C_32 = C_lmnj[(3,2,0)] + C_lmnj[(3,2,1)] * eta + C_lmnj[(3,2,2)] * eta**2

    # Storing QNM amplitudes C_{lmn} = C[(l,m)]
    C = {
        (2,1): C_21,
        (2,2): C_22,
        (3,2): C_32
    }


    # Storing C_2(-2,l',m',2,l'',-m') coefficients C_2(-2,l',m',2,l'',-m') = C2[(l,ll,m)]
    C2 = {
        (2,2,1): np.sqrt(5./np.pi)/14., #m=1
        (2,2,2): np.sqrt(5./np.pi)/7., #m=2
        (2,3,1): -np.sqrt(5./(14.*np.pi))/2., #m=1
        (2,3,2): 5./(4.*np.sqrt(7.*np.pi)), #m=2
        (2,3,3): 0., #m=3
        (3,2,1): -np.sqrt(5./(14.*np.pi))/2., #m=1
        (3,2,2): 5./(4.*np.sqrt(7.*np.pi)), #m=2
        (3,2,3): 0., #m=3
        (3,3,1): 0., #m=1
        (3,3,2): 0., #m=2
        (3,3,3): 0., #m=3
    }
    
    hmem_inf = np.zeros(len(t))
    for lp in [2,3]:
        for m in range(1, lp+1):
            for ldp in [2,3]:
                for lb in [2,3]:
                    for ldb in [2,3]:
                        for n in np.arange(0,8):
                            for nn in np.arange(0,8):
                                if m>lb or m>ldb or lp<m or ldp<m or (lb,m)==(3,1) or (ldb,m)==(3,1) or (lb,m)==(3,3) or (ldb,m)==(3,3):
                                    pass
                                else:
                                    hmem_inf += ((2/(np.sqrt(24)*mf)) * (-1)**m * (
                                        C2[(lp,ldp,m)] * C[(lb,m)][n] * np.conj(C[(ldb,m)][nn]) * A[(lp,lb,m)][n] * np.conj(A[(ldp,ldb,m)][nn])
                                           * (omega[(lb,m)][n] * np.conj(omega[(ldb,m)][nn]) / (omega[(lb,m)][n] - np.conj(omega[(ldb,m)][nn])))
                                           * np.exp(-1.j * (omega[(lb,m)][n] - np.conj(omega[(ldb,m)][nn])) *(t-ti)/mf)
                                    ).imag
                                                                     )
    
    return Delta_hmem_fit - hmem_inf


# ## Intermediate Memory and Full model

def compute_memory_model(q: float,
                         x: np.ndarray = x_int,
                         ti: float = -1e4,
                         tf: float = 1.3e2,
                         ti_int: float = -2e3,
                         tf_int: float = 0.,
                         dt: float = 0.01,
                         outputs: str = 'full'):
    
    '''
    Compute the full memory model
    
    Args:
        q : Mass ratio of the binary system (1 ≤ q ≤ 8)
        x : Array of optimized intermediate memory parameters
        ti : Start time of the intermediate memory
        tf : End time of the intermediate memory
        dt: Time step
        outputs: output options (full or components)
        
    Returns:
        dict of the time and full memory model, or
        dict of (dict of the time and full memory model,
                intermediate memory parameters,
                dict of the inspiral time and inspiral memory,
                dict of the intermediate time and intermediate memory,
                dict of the ringdown time and ringdown memory)

    '''
    p10, p11 = x[0:2]
    p20, p21 = x[2:4]
    p30, p31 = x[4:6]
    p40, p41 = x[6:8]
    p50, p51 = x[8:10]
    p60, p61 = x[10:12]
    c6 = x[12]
    
    # Compute symmetric mass ratio
    eta = q/(q+1)**2
    
    p1 = p10 + p11*eta
    p2 = p20 + p21*eta
    p3 = p30 + p31*eta
    p4 = p40 + p41*eta
    p5 = p50 + p51*eta
    p6 = p60 + p61*eta
    
    int_t = np.linspace(ti_int, tf_int, 1 + int(np.rint((tf_int - ti_int) / dt)))
    rd_t = np.linspace(tf_int, tf, 1 + int(np.rint((tf - tf_int) / dt)))
    insp_t = np.linspace(ti, ti_int, 1 + int(np.rint((ti_int - ti) / dt)))
    t = np.concatenate((insp_t[:-1], int_t, rd_t[1:]))
    
    hmem_insp,_ = compute_inspiral_memory_model(q=q, x=x_insp, ti=ti, tf=ti_int)
    hmem_rd= compute_ringdown_memory_model(q=q)[np.where(rd_t<=tf_int)[0][-1]:]
    
    hmem_insp_1d = np.gradient(hmem_insp, dt, edge_order=2)
    hmem_rd_1d = np.gradient(hmem_rd, dt, edge_order=2)
    
    hmem_insp_2d = np.gradient(hmem_insp_1d, dt, edge_order=2)
    hmem_rd_2d = np.gradient(hmem_rd_1d, dt, edge_order=2)

    Bs = np.vstack((np.array([1, np.exp(p1*tf_int), np.exp(p2*tf_int), np.exp(p3*tf_int), np.exp(p4*tf_int), np.exp(p5*tf_int)]),
                      np.array([1, np.exp(p1*ti_int), np.exp(p2*ti_int), np.exp(p3*ti_int), np.exp(p4*ti_int), np.exp(p5*ti_int)]),
                      np.array([0, p1*np.exp(p1*tf_int), p2*np.exp(p2*tf_int), p3*np.exp(p3*tf_int), p4*np.exp(p4*tf_int), p5*np.exp(p5*tf_int)]),
                      np.array([0, p1*np.exp(p1*ti_int), p2*np.exp(p2*ti_int), p3*np.exp(p3*ti_int), p4*np.exp(p4*ti_int), p5*np.exp(p5*ti_int)]),
                      np.array([0, p1**2 * np.exp(p1*tf_int), p2**2 * np.exp(p2*tf_int), p3**2 * np.exp(p3*tf_int), p4**2 * np.exp(p4*tf_int), p5**2 * np.exp(p5*tf_int) ]),
                      np.array([0, p1**2 * np.exp(p1*ti_int), p2**2 * np.exp(p2*ti_int), p3**2 * np.exp(p3*ti_int), p4**2 * np.exp(p4*ti_int), p5**2 * np.exp(p5*ti_int) ]) ))

    Bs_inv = np.linalg.pinv(Bs)
    Bf = np.reshape(np.array([np.exp(p6*tf_int), np.exp(p6*ti_int), p6*np.exp(p6*tf_int), p6*np.exp(p6*ti_int), p6**2 * np.exp(p6*tf_int), p6**2 * np.exp(p6*ti_int)]),(6,1))
    cf = np.reshape(c6,(1,1))
    A = np.reshape(np.array([hmem_rd[0], hmem_insp[-1], hmem_rd_1d[0], hmem_insp_1d[-1], hmem_rd_2d[0], hmem_insp_2d[-1]]),(6,1))
    A_new = np.reshape(A,(6,1))-np.dot(Bf,cf)
    cs = np.dot(Bs_inv, A_new)

    c0, c1, c2, c3, c4, c5 = np.reshape(cs, (1,6))[0]

    hmem_int = c0 + c1*np.exp(p1*int_t) + c2*np.exp(p2*int_t) + c3*np.exp(p3*int_t) + c4*np.exp(p4*int_t) + c5*np.exp(p5*int_t) + c6*np.exp(p6*int_t) 

    hmem = np.concatenate((hmem_insp[:-1], hmem_int, hmem_rd[1:]))
    int_params = np.asarray([c0, c1, c2, c3, c4, c5, c6, p1, p2, p3, p4, p5, p6])
    
    if outputs == 'full':
        return {'time': t,
                'memory': hmem}
    elif outputs == 'components':
        return {'full_memory': {'time': t,
                                'memory': hmem},
                'int_memory_params': int_params,
                'insp_memory': {'time': insp_t,
                                'memory': hmem_insp},
                'int_memory': {'time': int_t,
                               'memory': hmem_int},
                'rd_memory': {'time': rd_t,
                              'memory': hmem_rd}
               }
    else:
        return 'Enter a valid output variable option: full or components'

