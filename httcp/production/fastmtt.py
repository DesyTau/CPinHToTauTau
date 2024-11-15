import math
import functools

from typing import Optional
from columnflow.util import maybe_import

import math
np = maybe_import("numpy")
#nb = maybe_import("numba")
#root = maybe_import("ROOT")

# custom LorentzVector class
class lorentzvector:
    def __init__(self,
                 pt: np.float32,
                 eta: np.float32,
                 phi: np.float32,
                 mass: np.float32):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.mass = mass
        self.px = self.pt*math.cos(self.phi)
        self.py = self.pt*math.sin(self.phi)
        self.pz = self.pt*math.sinh(self.eta)
        self.E = math.sqrt(self.px*self.px+self.py*self.py+self.pz*self.pz+self.mass*self.mass)

#@nb.jit(nopython=True, parallel=False)
def fastmtt(pt_1, eta_1, phi_1, mass_1, decay_type_1,
            pt_2, eta_2, phi_2, mass_2, decay_type_2,
            met_x, met_y, metcov_xx, metcov_xy, metcov_yx, metcov_yy,
            verbosity=-1,
            delta=1/1.15,
            reg_order=6,
            mH = 125.10,
            GammaH = 0.004,
            mass_lower = 40.0,
            mass_upper = 200.0):

    # initialize global parameters
    m_ele = 0.51100e-3
    m_muon = 0.10566
    m_tau = 1.77685
    m_pion = 0.13957
    mass_dict = {0: m_ele, 1: m_muon}

#    test_mass_min = mH - margin
#    test_mass_max = mH + margin

    N = len(pt_1)
    # initialize outputs higgs->ditau decays
    pt_1_tt = np.zeros(N, dtype=np.float32)
    pt_2_tt = np.zeros(N, dtype=np.float32)
    pt_1_tt_BW = np.zeros(N, dtype=np.float32)
    pt_2_tt_BW = np.zeros(N, dtype=np.float32)

    # loop over all events, calculate corrected ditau mass
    for i in range(N):

        if (i%1000): print('fastmtt : processed events %i'%i)
        
        # grab the correct masses based on tau decay type
        # tau decay_type: 0 ==> leptonic to electron, 
        #                 1 ==> leptonic to muon, 
        #                 2 ==> leptonic to hadronic
        if (decay_type_1[i]==0): m1 = m_ele
        elif (decay_type_1[i]==1): m1 = m_muon
        else: m1 = mass_1[i]
        if (decay_type_2[i]==0): m2 = m_ele
        elif (decay_type_2[i]==1): m2 = m_muon
        else: m2 = mass_2[i]
            
        # store visible masses
        m_vis_1 = m1
        m_vis_2 = m2
        
        # determine minimum and maximum possible masses
        m_vis_min_1, m_vis_max_1 = 0, 0
        m_vis_min_2, m_vis_max_2 = 0, 0
        if (decay_type_1[i] == 0): m_vis_min_1, m_vis_max_1 = m_ele, m_ele
        if (decay_type_1[i] == 1): m_vis_min_1, m_vis_max_1 = m_muon, m_muon
        if (decay_type_1[i] == 2): m_vis_min_1, m_vis_max_1 = m_pion, 1.5
        if (decay_type_2[i] == 0): m_vis_min_2, m_vis_max_2 = m_ele, m_ele
        if (decay_type_2[i] == 1): m_vis_min_2, m_vis_max_2 = m_muon, m_muon
        if (decay_type_2[i] == 2): m_vis_min_2, m_vis_max_2 = m_pion, 1.5
        if (m_vis_1 < m_vis_min_1): m_vis_1 = m_vis_min_1
        if (m_vis_1 > m_vis_max_1): m_vis_1 = m_vis_max_1
        if (m_vis_2 < m_vis_min_2): m_vis_2 = m_vis_min_2
        if (m_vis_2 > m_vis_max_2): m_vis_2 = m_vis_max_2

        # store both tau candidate four vectors
        leg1 = lorentzvector(pt_1[i],eta_1[i],phi_1[i],m_vis_1)
        leg2 = lorentzvector(pt_2[i],eta_2[i],phi_2[i],m_vis_2)
        # avoiding lorentzvectors from ROOT and akward
        px_vis = leg1.px + leg2.px
        py_vis = leg1.py + leg2.py
        pz_vis = leg1.pz + leg2.pz
        E_vis  = leg1.E + leg2.E
        p_vis = math.sqrt(px_vis*px_vis+
                          py_vis*py_vis+
                          pz_vis*pz_vis)
        m_vis = math.sqrt(E_vis*E_vis-p_vis*p_vis)
        
        # correct initial visible masses
        if (decay_type_1[i] == 2 and m_vis_1 > 1.5): m_vis_1 = 0.3
        if (decay_type_2[i] == 2 and m_vis_2 > 1.5): m_vis_2 = 0.3

        # invert met covariance matrix, calculate determinant
        metcovinv_xx, metcovinv_yy = metcov_yy[i], metcov_xx[i]
        metcovinv_xy, metcovinv_yx = -metcov_xy[i], -metcov_yx[i]
        metcovinv_det = (metcovinv_xx*metcovinv_yy -
                         metcovinv_yx*metcovinv_xy)
        if (metcovinv_det<1e-10): 
                print("Warning! Ill-conditioned MET covariance at event index", i)
                continue
               
        # perform likelihood scan 
        # see http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_032_v3.pdf
        met_const = 1/(2*math.pi*math.sqrt(metcovinv_det))
        min_likelihood, x1_opt, x2_opt = 999999, 0.02, 0.02
        min_likelihood_BW, x1_opt_BW, x2_opt_BW = 999999, 0.02, 0.02 
        mass_likelihood, met_transfer = 0, 0
        
        # scan over weights for each ditau four-vector
        initialise = True
        for x1 in np.arange(0.05, 0.95, 0.05):
            for x2 in np.arange(0.05, 0.95, 0.05):
                test_mass = m_vis/math.sqrt(x1*x2)
                if (test_mass>mass_upper or test_mass<mass_lower):
                    continue
#                if (test_mass<test_mass_min or test_mass>test_mass_max):
#                    continue
                x1_min = min(1, math.pow((m_vis_1/m_tau),2))
                x2_min = min(1, math.pow((m_vis_2/m_tau),2))
                if ((x1 < x1_min) or (x2 < x2_min)): 
                    continue
        
                # test weighted four-vectors
                nu_test_px = leg1.px/x1 + leg2.px/x2 - px_vis
                nu_test_py = leg1.py/x1 + leg2.py/x2 - py_vis

                # calculate mass likelihood integral 
                m_shift = test_mass * delta
                if (m_shift < m_vis): continue 
                x1_min = min(1.0, math.pow((m_vis_1/m_tau),2))
                x2_min = max(math.pow((m_vis_2/m_tau),2), 
                             math.pow((m_vis/m_shift),2))
                x2_max = min(1.0, math.pow((m_vis/m_shift),2)/x1_min)
                if (x2_max < x2_min): continue
                J = 2*math.pow(m_vis,2) * math.pow(m_shift, -reg_order)
                I_x2 = math.log(x2_max) - math.log(x2_min)
                I_tot = I_x2
                if (decay_type_1[i] != 2):
                    I_m_nunu_1 = math.pow((m_vis/m_shift),2) * (math.pow(x2_max,-1) - math.pow(x2_min,-1))
                    I_tot += I_m_nunu_1
                if (decay_type_2[i] != 2):
                    I_m_nunu_2 = math.pow((m_vis/m_shift),2) * I_x2 - (x2_max - x2_min)
                    I_tot += I_m_nunu_2
                mass_likelihood = 1e9 * J * I_tot
                
                # calculate MET transfer function 
                residual_x = met_x[i] - nu_test_px
                residual_y = met_y[i] - nu_test_py
                pull2 = (residual_x*(metcovinv_xx*residual_x + 
                                     metcovinv_xy*residual_y) +
                         residual_y*(metcovinv_yx*residual_x +
                                     metcovinv_yy*residual_y))
                pull2 /= metcovinv_det
                met_transfer = met_const*math.exp(-0.5*pull2)
                
                # calculate final likelihood, store if minimum
                deltaM = test_mass*test_mass-mH*mH
                mG = test_mass*GammaH
                BreitWigner_likelihood = 1/(deltaM*deltaM + mG*mG) 
                likelihood = -met_transfer * mass_likelihood
                likelihood_BW = likelihood*BreitWigner_likelihood
                if initialise:
                    min_likelihood = likelihood
                    min_likelihood_BW = likelihood_BW
                    x1_opt, x2_opt = x1, x2
                    x1_opt_BW, x2_opt_BW = x1, x2
                    initialise = False
                else:
                    if (likelihood < min_likelihood):
                        min_likelihood = likelihood
                        x1_opt, x2_opt = x1, x2
                    if (likelihood_BW < min_likelihood_BW):
                        min_likelihood_BW = likelihood_BW
                        x1_opt_BW, x2_opt_BW = x1, x2
                    
        pt_1_tt[i]   = pt_1[i]/x1_opt
        pt_2_tt[i]   = pt_2[i]/x2_opt
        pt_1_tt_BW[i]   = pt_1[i]/x1_opt_BW
        p2_1_tt_BW[i]   = pt_2[i]/x2_opt_BW
        

    return {'pt_fast_lep0': pt_1_tt,
            'pt_fast_lep1': pt_2_tt,
            'pt_cons_lep0': pt_1_tt_BW,
            'pt_cons_lep1': pt_2_tt_BW}





