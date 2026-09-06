#!/usr/bin/env python

# This code implements the waveform model developed in the following two papers:
# Paper I: Arwa Elhashash and David A. Nichols. ``Waveform models for the gravitational-wave memory effect: Extreme mass-ratio limit and final memory offset''. Phys. Rev. D 111, 044052 (2025). ArXiv:2407.19017.
# Paper II: Arwa Elhashash and David A. Nichols. ``Waveform models for the gravitational-wave memory effect: II. Time-domain and frequency-domain models for nonspinning binaries''. Phys. Rev. D 112, 064014 (2025). ArXiv:2504.18635.

# Needed packages
import numpy as np
import qnm
import surfinBH
from typing import Tuple

# Loading the remnant fit for final mass and spin calculation
remnant_fit = surfinBH.LoadFits('NRSur3dq8Remnant')

# ## Time-Domain Model Parameters

# Coefficients for the inspiral memory model (Table III. in paper II)
x_insp = np.asarray([2155.81002984, -22124.15922533, 117604.79338105, -331057.67271698, 380724.41433229])

# Coefficients for the intermediate memory model (Table IV. in paper II)
x_int = np.asarray([8.49306e-1, 1.02024e1, 4.29098e-2, 2.36601e-1, 8.37061e-3, 1.59611e-2, 5.97250e-4, 1.10423e-3, 8.23572e-1, 1.01875e0, 9.40991e-4, 6.22363e-3, 1.19119e-3])

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

# ## Frequency-Domain Memory Model Parameters

# Amplitude-model coefficients:
# x = [a11, a12, a21, a22, a32, b10, b11, b20, b21, b30, b31, c0, c1]
# (a31 is fixed to zero by construction; see compute_global_amplitude_model)
amplitude_params = np.asarray([-4.62939930e-01,  8.05972250e-01, -1.93002245e+00, -7.92143357e-01,
        1.09945379e+00,  1.80967806e+00,  1.97447613e-04,  4.64415781e+00,
        2.36292617e+00,  1.82272214e+00,  1.61541350e-05,  9.99923604e-01,
        9.49905432e-01])

# Phase-model coefficients:
# x = [a0, a1, a2,  b0, b1, b2,  c0, c1, c2,  d0, d1, d2,  g0, g1, g2,  h0, h1, h2]
phase_params = np.asarray([ 1.54644804e+02, -3.83187952e+02, -1.09433432e+02, -4.53639279e-02,
        1.96682245e+00, -6.42498632e+00, -4.90466459e+00,  4.66409968e+02,
        5.54509756e+02, -4.90706383e-02,  1.42544528e+00, -2.91977247e+00,
        6.54194457e+00, -1.30731611e+02,  7.91180733e+02, -2.92996886e-01,
        6.37110873e+00, -1.00876756e+01])

# ## Time-Domain Memory Model
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

def compute_TD_memory_model(q: float,
                            x: np.ndarray = x_int,
                            ti: float = -1e4,
                            tf: float = 1.3e2,
                            ti_int: float = -2e3,
                            tf_int: float = 2.,
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
    # Extract parameters 
    params = x[:12].reshape(6, 2) # First 12 elements
    c6 = x[12]
    
    # Compute symmetric mass ratio
    eta = q / (q + 1)**2
    
    # Compute eta-dependent parameters
    p_values = params[:, 0] + params[:, 1] * eta
    p1, p2, p3, p4, p5, p6 = p_values
    
    # Create time arrays
    int_t = np.linspace(ti_int, tf_int, 1 + int(np.rint((tf_int - ti_int) / dt)))
    rd_t = np.linspace(tf_int, tf, 1 + int(np.rint((tf - tf_int) / dt)))
    insp_t = np.linspace(ti, ti_int, 1 + int(np.rint((ti_int - ti) / dt)))
    
    # Concatenate time arrays, avoiding duplicates at boundaries
    t = np.concatenate((insp_t[:-1], int_t, rd_t[1:]))
    
    # Compute memory components
    hmem_insp,_ = compute_inspiral_memory_model(q=q, x=x_insp, ti=ti, tf=ti_int, dt=dt)
    hmem_rd= compute_ringdown_memory_model(q=q, ti=tf_int, dt=dt)
    
    # Compute derivatives and edge handling
    hmem_insp_1d = np.gradient(hmem_insp, dt, edge_order=2)
    hmem_rd_1d = np.gradient(hmem_rd, dt, edge_order=2)
    
    hmem_insp_2d = np.gradient(hmem_insp_1d, dt, edge_order=2)
    hmem_rd_2d = np.gradient(hmem_rd_1d, dt, edge_order=2)

    # Build coefficient matrix more systematically
    p_array = np.array([p1, p2, p3, p4, p5])
    
    # Exponential terms at boundary points
    exp_tf = np.exp(p_array * tf_int)
    exp_ti = np.exp(p_array * ti_int)
    
    # Build the 6x6 coefficient matrix
    Bs = np.array([
        [1, *exp_tf],                           # Function values at tf_int
        [1, *exp_ti],                           # Function values at ti_int
        [0, *(p_array * exp_tf)],               # First derivatives at tf_int
        [0, *(p_array * exp_ti)],               # First derivatives at ti_int
        [0, *(p_array**2 * exp_tf)],            # Second derivatives at tf_int
        [0, *(p_array**2 * exp_ti)]             # Second derivatives at ti_int
    ])

    Bs_inv = np.linalg.pinv(Bs)
    
    # Build p6 contribution vector
    exp_p6_tf = np.exp(p6 * tf_int)
    exp_p6_ti = np.exp(p6 * ti_int)
    
    Bf = np.array([
        exp_p6_tf,
        exp_p6_ti,
        p6 * exp_p6_tf,
        p6 * exp_p6_ti,
        p6**2 * exp_p6_tf,
        p6**2 * exp_p6_ti
    ]).reshape(6, 1)
    
    # Boundary conditions from inspiral and ringdown
    boundary_values = np.array([
        hmem_rd[0], hmem_insp[-1],
        hmem_rd_1d[0], hmem_insp_1d[-1],
        hmem_rd_2d[0], hmem_insp_2d[-1]
    ]).reshape(6, 1)
    
    # Solve for c0-c5 coefficients
    A_adjusted = boundary_values - Bf * c6
    cs = np.dot(Bs_inv, A_adjusted).flatten()
    c0, c1, c2, c3, c4, c5 = cs

    # Compute intermediate memory
    exp_terms = np.array([np.exp(p_val * int_t) for p_val in p_array])
    hmem_int = (c0 + 
                np.sum(cs[1:6] * exp_terms.T, axis=1) + 
                c6 * np.exp(p6 * int_t))
    
    # Combine all memory components
    hmem = np.concatenate((hmem_insp[:-1], hmem_int, hmem_rd[1:]))
    
    # Intermediate parameters
    int_params = np.asarray([c0, c1, c2, c3, c4, c5, c6, p1, p2, p3, p4, p5, p6])
    
    # Return results based on output type
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

# ## Frequency-Domain Memory Model
# ## Amplitude Model

def compute_global_amplitude_model(x: np.ndarray,
                                   q: float,
                                   freq: np.ndarray):

    """
    Evaluate the frequency-domain memory amplitude model.
 
    The amplitude is a sum of three "csch" basis functions, each with a
    coefficient (a1, a2, a3) and a width parameter (b1, b2, b3) that
    depend on the symmetric mass ratio eta = q / (1 + q)^2. The
    coefficients a_i are quadratic functions of eta:
 
        a_i = a_i1 * eta + a_i2 * eta^2
 
    with the constraint a31 = 0 (i.e. a3 = a32 * eta^2 only).
 
    Parameters
    ----------
    x : Length-13 array of fitted coefficients, ordered as
        [a11, a12, a21, a22, a32, b10, b11, b20, b21, b30, b31, c0, c1].
    q : Binary mass ratio, q = m1 / m2 >= 1.
    freq : Array of (dimensionless) frequencies at which to evaluate the
        amplitude model.
 
    Returns
    -------
    An array of the frequency-domain memory amplitude model, h * freq (i.e. this
        must be divided by `freq` to get the strain amplitude; see
        `compute_hmem_model`).
    """
    
    # Unpack coefficients (each b/a coefficient is stored as log10 in x)
    a11, a12 = 10**x[0],  10**x[1]
    a21, a22 = 10**x[2],  10**x[3]
    a32      = 10**x[4]               # a31 = 0 enforced

    b10, b11 = 10**x[5],  10**x[6]
    b20, b21 = 10**x[7],  10**x[8]
    b30, b31 = 10**x[9],  10**x[10]

    c0, c1   = x[11], x[12]

    # Symmetric mass ratio
    eta = q / (1 + q) ** 2
 
    # Mass-ratio-dependent coefficients
    a1 = a11 * eta + a12 * eta**2
    a2 = a21 * eta + a22 * eta**2
    a3 =             a32 * eta**2     # a31 = 0

    b1 = b10 + b11 * eta
    b2 = b20 + b21 * eta
    b3 = b30 + b31 * eta

    c  = c0  + c1  * eta

    # Sum of three basis functions
    hmem_amp_model = (  a1 * np.pi / np.sinh(b1 * freq * np.pi / 2)
                      + a2 * np.pi / np.sinh(b2 * freq * np.pi / 2)
                      - a3 * freq**(c-1)    * np.pi / np.sinh(b3 * freq * np.pi / 2))

    return hmem_amp_model


# ## Phase Model

def compute_global_phase_model(x: np.ndarray,
                              q: float,
                              freq: np.ndarray,
                              tf: float = 1.3e2):
    
    """
    Evaluate the frequency-domain memory phase model.
 
    The phase is built from three basis functions (two exponential decays
    and a power law), each with a coefficient that is a quadratic function
    of the symmetric mass ratio eta = q / (1 + q)^2:
 
        x = x_0 + eta * x_1 + eta^2 * x_2
    The 2*pi*tf*freq term aligns the model with the peak-at-t=0 convention.
   
    Parameters
    ----------
    x : Length-18 array of fitted coefficients, ordered as
        [a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2,
         g0, g1, g2, h0, h1, h2].
    q : Binary mass ratio, q = m1 / m2 >= 1.
    freq : Array of (dimensionless) frequencies at which to evaluate the
        phase model.
    tf : Reference/end time (in M) of the time-domain model used when this
        phase fit was performed. Defaults to 1.3e2; only override this
        if you are deliberately testing a different alignment.
  
    Returns
    -------
    An array of the frequency-domain memory phase model (radians).
    """

    a0, a1, a2 = x[0:3]
    b0, b1, b2 = x[3:6]
    c0, c1, c2 = x[6:9]
    d0, d1, d2 = x[9:12]
    g0, g1, g2 = x[12:15]
    h0, h1, h2 = x[15:]
    
    eta = q/(q+1)**2
    a = a0 + eta * a1 + eta**2 * a2
    b = b0 + eta * b1 + eta**2 * b2
    c = c0 + eta * c1 + eta**2 * c2
    d = d0 + eta * d1 + eta**2 * d2
    g = g0 + eta * g1 + eta**2 * g2
    h = h0 + eta * h1 + eta**2 * h2

    hmem_phase_model = (-a * freq * np.exp(-b/freq)
                        - c * freq * np.exp(-freq/d) 
                        + g * freq**h
                        - np.pi/2
                        + 2*np.pi*tf*freq)

    return hmem_phase_model


# ## Frequency-domain Full Model

def compute_FD_memory_model(q: float,
                           freq: np.ndarray,
                           amp_x: np.ndarray = amplitude_params,
                           ph_x: np.ndarray = phase_params):
    
    """
    Compute the full complex frequency-domain memory waveform, h_mem(f).
 
    Combines the amplitude and phase models into
 
        h_mem(f) = A(f) * exp(i * Phi(f))
 
    Parameters
    ----------
    q : Binary mass ratio, q = m1 / m2 >= 1.
    freq : Array of (dimensionless) frequencies at which to evaluate the model.
    amp_x : Fitted amplitude-model coefficients (see `amplitude_params`).
    ph_x : Fitted phase-model coefficients (see `phase_params`).
 
    Returns
    -------
    Complex array of the frequency-domain memory strain, hmem_model(f).
    """
    
    amp_model = compute_global_amplitude_model(x=amp_x, q=q, freq=freq)
    phase_model = compute_global_phase_model(q=q, x=ph_x, freq=freq)
    
    hmem_model = amp_model * np.exp(1.j*phase_model)
    
    return hmem_model
