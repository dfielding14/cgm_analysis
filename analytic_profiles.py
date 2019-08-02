import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
import matplotlib
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = "round"
matplotlib.rcParams['lines.solid_capstyle'] = "round"
import matplotlib.pyplot as plt
import glob
import numpy as np
import h5py
from scipy import interpolate
from scipy import optimize
from scipy import integrate
import scipy
from matplotlib import rc
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint
import scipy, numpy as np
from scipy import integrate, interpolate
from numpy import log as ln, log10 as log, pi ## note i'm using log for log10!! forgive me that's how i was raised
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import colossus, colossus.cosmology.cosmology
colossus.cosmology.cosmology.setCosmology('planck15')
from colossus.halo import profile_dk14

gamma   = 5/3.
kb      = 1.3806488e-16
mp      = 1.67373522381e-24
km      = 1e5
s       = 1
yr      = 3.1536e7
Myr     = 3.1536e13
Gyr     = 3.1536e16
pc      = 3.086e18
kpc     = 1.0e3 * pc
Mpc     = 1.0e6 * pc
H0      = 70*km/s/Mpc
G       = 6.673e-8
Msun    = 2.e33
OL      = 0.73
Om      = 0.27
fb      = 0.158
keV     = 1.60218e-9

mu = 0.62
metallicity = 10**-0.5
muH = 1/0.75
redshift=0.0


def c_DuttonMaccio14(lMhalo, z=0):  #table 3 appropriate for Mvir
    c_z0  = lambda lMhalo: 10.**(1.025 - 0.097*(lMhalo-log(0.7**-1*1e12))) 
    c_z05 = lambda lMhalo: 10.**(0.884 - 0.085*(lMhalo-log(0.7**-1*1e12))) 
    c_z1  = lambda lMhalo: 10.**(0.775 - 0.073*(lMhalo-log(0.7**-1*1e12))) 
    c_z2  = lambda lMhalo: 10.**(0.643 - 0.051*(lMhalo-log(0.7**-1*1e12)))
    zs = np.array([0.,0.5,1.,2.])
    cs = np.array([c_func(lMhalo) for c_func in (c_z0,c_z05,c_z1,c_z2)])
    return np.interp(z, zs, cs)
def Behroozi_params(z, parameter_file='./data/smhm_true_med_cen_params.txt'):
    param_file = open(parameter_file, "r")
    param_list = []
    allparams = []
    for line in param_file:
        param_list.append(float((line.split(" "))[1]))
        allparams.append(line.split(" "))
    
    if (len(param_list) != 20):
        print("Parameter file not correct length.  (Expected 20 lines, got %d)." % len(param_list))
        quit()
    
    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ");
    params = dict(zip(names, param_list))
    
    
    #Print SMHM relation
    a = 1.0/(1.0+z)
    a1 = a - 1.0
    lna = np.log(a)
    zparams = {}
    zparams['m_1'] = params['M_1'] + a1*params['M_1_A'] - lna*params['M_1_A2'] + z*params['M_1_Z']
    zparams['sm_0'] = zparams['m_1'] + params['EFF_0'] + a1*params['EFF_0_A'] - lna*params['EFF_0_A2'] + z*params['EFF_0_Z']
    zparams['alpha'] = params['ALPHA'] + a1*params['ALPHA_A'] - lna*params['ALPHA_A2'] + z*params['ALPHA_Z']
    zparams['beta'] = params['BETA'] + a1*params['BETA_A'] + z*params['BETA_Z']
    zparams['delta'] = params['DELTA']
    zparams['gamma'] = 10**(params['GAMMA'] + a1*params['GAMMA_A'] + z*params['GAMMA_Z'])
    
    smhm_max = 14.5-0.35*z
    if (params['CHI2']>200):
        print('#Warning: chi^2 > 200 implies that not all features are well fit.  Comparison with the raw data (in data/smhm/median_raw/) is crucial.')
    ms = 0.05 * np.arange(int(10.5*20),int(smhm_max*20+1),1)
    dms = ms - zparams['m_1'] 
    dm2s = dms/zparams['delta']
    sms = zparams['sm_0'] - log(10**(-zparams['alpha']*dms) + 10**(-zparams['beta']*dms)) + zparams['gamma']*np.e**(-0.5*(dm2s*dm2s))
    return ms,sms

def MgalaxyBehroozi(lMhalo, z, parameter_file='./data/smhm_true_med_cen_params.txt'):
    ms,sms = Behroozi_params(z,parameter_file)
    lMstar = scipy.interpolate.interp1d(ms, sms, fill_value='extrapolate')(lMhalo)
    return 10.**lMstar*un.Msun


class DK14_with_Galaxy:
    mu=0.6
    X=0.75    
    gamma = 5/3.
    def __init__(self,Mgalaxy,half_mass_radius=None,dk14=None,**kwargs):
        if dk14!=None: 
            self.dk14=dk14
        else:
            self.dk14=colossus.halo.profile_dk14.getDK14ProfileWithOuterTerms(**kwargs)        
        self.Mgalaxy = Mgalaxy
        self.z = kwargs['z']
        if half_mass_radius==None: self.half_mass_radius = 0.015 * self.RDelta('200c') #Kravtsov 2013
        else: self.half_mass_radius = half_mass_radius

        self._Rs = 10.**np.arange(-4.,1.5,0.01) * self.rvir().to('kpc').value
        self._Ms = self.dk14.enclosedMass(self._Rs*cosmo.h) / cosmo.h  #only DM mass
        self._Ms += self.enclosedMass_galaxy(self._Rs*un.kpc).to('Msun').value
        drs = (self._Rs[2:]-self._Rs[:-2])/2.
        drs = np.pad(drs,1,mode='edge')
        self._phis = ((-self.g(self._Rs*un.kpc)[::-1].to('km**2*s**-2*kpc**-1').value * drs[::-1]).cumsum())[::-1]        
        
    def enclosedMass_galaxy(self,r):
        return self.Mgalaxy * r/(r+self.half_mass_radius)
    def enclosedMass(self,r):
        return np.interp(r.to('kpc').value, self._Rs, self._Ms)*un.Msun
    def enclosedMassInner(self,r):
        return self.dk14.enclosedMassInner(r.to('kpc').value*cosmo.h)*un.Msun / cosmo.h  #only DM mass
    def enclosedMassOuter(self,r):
        return self.dk14.enclosedMassOuter(r.to('kpc').value*cosmo.h)*un.Msun / cosmo.h  #only DM mass
    def g(self, r):
        return cons.G*self.enclosedMass(r) / r**2  
    def rvir(self):
        return self.dk14.RDelta(self.z,'vir') * un.kpc/cosmo.h
    def RDelta(self,mdef):
        return self.dk14.RDelta(self.z,mdef) * un.kpc/cosmo.h
    def Tvir(self):
        return (self.mu * cons.m_p * self.vc(self.rvir())**2 / (2*cons.k_B)).to('K')    
    def vc(self,r):
        return ((cons.G*self.enclosedMass(r) / r)**0.5).to('km/s')
    def Tc(self,r):
        return (self.mu*cons.m_p * self.vc(r)**2 / (self.gamma*cons.k_B)).to('K')
    def Mhalo(self):
        return self.dk14.MDelta(self.z,'vir')*un.Msun/cosmo.h
    def phi(self,rs,r0):
        phis = np.interp(rs.to('kpc').value, self._Rs, self._phis)
        phi0 = np.interp(r0.to('kpc').value, self._Rs, self._phis)        
        return (phis - phi0) * un.km**2/un.s**2
    def rho(self,r):
        return (self.dk14.density(r.to('kpc').value*cosmo.h) * un.Msun * cosmo.h**2 / un.kpc**3).to('g*cm**-3')
    def tff(self,r):
        return (2**0.5 * r/ self.vc(r)).to('Gyr')
    def rho_b(self,r):
        return self.rho(r) * cosmo.Ob0 / cosmo.Om0



"""
Cooling curve as a function of density, temperature, metallicity, redshift
"""
file = glob.glob('./data/Cooling_Tables/Lambda_tab.npz')
if len(file) > 0:
    data = np.load(file[0])
    Lambda_tab = data['Lambda_tab']
    redshifts  = data['redshifts']
    Zs         = data['Zs']
    log_Tbins  = data['log_Tbins']
    log_nHbins = data['log_nHbins']    
    Lambda      = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs,redshifts), Lambda_tab, bounds_error=False, fill_value=0)
else:
    files = np.sort(glob.glob('./data/Cooling_Tables/z_*hdf5'))
    redshifts = np.array([float(f[-10:-5]) for f in files])
    HHeCooling = {}
    ZCooling   = {}
    TE_T_n     = {}
    for i in xrange(len(files)):
        f            = h5py.File(files[i], 'r')
        i_X_He       = -3 
        Metal_free   = f.get('Metal_free')
        Total_Metals = f.get('Total_Metals')
        log_Tbins    = np.array(np.log10(Metal_free['Temperature_bins']))
        log_nHbins   = np.array(np.log10(Metal_free['Hydrogen_density_bins']))
        Cooling_Metal_free       = np.array(Metal_free['Net_Cooling'])[i_X_He] ##### what Helium_mass_fraction to use    Total_Metals = f.get('Total_Metals')
        Cooling_Total_Metals     = np.array(Total_Metals['Net_cooling'])
        HHeCooling[redshifts[i]] = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Metal_free)
        ZCooling[redshifts[i]]   = interpolate.RectBivariateSpline(log_Tbins,log_nHbins, Cooling_Total_Metals)
        f.close()
    Lambda_tab  = np.array([[[[HHeCooling[zz].ev(lT,ln)+Z*ZCooling[zz].ev(lT,ln) for zz in redshifts] for Z in Zs] for lT in log_Tbins] for ln in log_nHbins])
    np.savez('./data/Cooling_Tables/Lambda_tab.npz', Lambda_tab=Lambda_tab, redshifts=redshifts, Zs=Zs, log_Tbins=log_Tbins, log_nHbins=log_nHbins)
    Lambda      = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs,redshifts), Lambda_tab, bounds_error=False, fill_value=0)
print("interpolated lambda")


lMhalo=12.
nu =  0.845
f_cgm = 0.01 

"""
gamma P / rho = cs^2 

vc^2 / cs^2 = fcs
"""
Mhalo = 10**lMhalo * Msun
z = 0.
Mgalaxy=MgalaxyBehroozi(lMhalo, z)
dk14 = DK14_with_Galaxy(Mgalaxy=Mgalaxy,z=z,M = 10.**lMhalo*cosmo.h, c = c_DuttonMaccio14(lMhalo,z), mdef = 'vir')
cnfw = c_DuttonMaccio14(lMhalo,z)
rvir = dk14.rvir().value
r200m = dk14.RDelta('200m').value
rgal = dk14.half_mass_radius.value

print ('Mhalo  = %e' % 10**lMhalo)
print ('cnfw  = %f' % cnfw)
print ('rvir  = %f' % rvir)
print ('r200m  = %f' % r200m)
print ('Mgal  = %e' % Mgalaxy.value)
print ('Rgal  = %f' % rgal)

H0 = 67.74*km/s/Mpc
Om = 0.3075
rhom = (3 * H0**2 * Om * (1.+z)**3) / (8*np.pi*G)
rhoc = (3 * H0**2) / (8*np.pi*G)
rs = rvir/cnfw * kpc
rho0 = Mhalo / (4 * np.pi * rs**3 * ( np.log(1.+cnfw) - cnfw/(1.+cnfw) ))
rt = (1.9-0.18*nu)*r200m * kpc
a = 5. * cnfw * r200m / rvir
b = rs/rt

def grav_acc(r):
    # 1/r**2 d/dr(  r**2 g ) = 4 pi G rho
    # g(r) = integral ( 4 pi G rho_NFW r**2 ) / r**2
    # g_NFW(r) = integral ( 4 pi G rhos rs**2 x**2/(x*(1+x)**2)) / (rs**2  x**2 )
    # g_NFW(r) = 4 pi G rhos (1/(x+1) + log(x+1))|_0**x /x**2
    # g_NFW(r) = 4 pi G rhos (log(x+1)-x/(x+1)) /x**2


    # rho_DM = rhos / (x * (1+x)**2 ) / (1.0 + (rs/rt)**4  * x**4)**2 
    #        + rhom * ( (rs/5*rvir)**-1.5 * x**-1.5 + 1. )

    # rho_DM = rhos / (x * (1+x)**2 ) / (1.0 + b**4  * x**4)**2 
    #        + rhom * ( bb**-1.5 * x**-1.5 + 1. )
    x = r/rs
    g = 4. * np.pi * G * rs
    g *= ((64.*a**1.5*rhom*x**1.5 + 32.*rhom*x**3. + (96.*rho0)/((1. + b**4.)**2.*(1. + x)) - 
         (24.*rho0*(-1. + b**4.*(3. + x*(-4. + x*(3. - 2.*x + b**4.*(-1. + 2.*x))))))/((1. + b**4.)**2.*(1. + b**4.*x**4.)) + 
         (12.*b*(-5.*np.sqrt(2.) + b*(18. - 14.*np.sqrt(2.)*b + 12.*np.sqrt(2.)*b**3. - 16.*b**4. + 2.*np.sqrt(2.)*b**5. + np.sqrt(2.)*b**7. - 2.*b**8.))*rho0*
            np.arctan(1. - np.sqrt(2.)*b*x))/(1. + b**4.)**3. + 
         (6*rho0*(4.*(1. + b**4.)*(-5. + 3.*b**4.) + 2.*b**2.*(-9 + 8.*b**4. + b**8.)*np.pi - 
              2.*b*(-5.*np.sqrt(2.) + b*(-18. - 14.*np.sqrt(2.)*b + 12.*np.sqrt(2.)*b**3. + 16.*b**4. + 2.*np.sqrt(2.)*b**5. + np.sqrt(2.)*b**7. + 2.*b**8.))*
               np.arctan(1. + np.sqrt(2.)*b*x) + 16.*(1. - 7.*b**4.)*np.log(1. + x) + 4.*(-1. + 7.*b**4.)*np.log(1. + b**4.*x**4.) - 
              np.sqrt(2.)*b*(-5. + 14.*b**2. + 12.*b**4. - 2.*b**6 + b**8.)*(np.log(1. + b*x*(-np.sqrt(2.) + b*x)) - np.log(1. + b*x*(np.sqrt(2.) + b*x)))))/
          (1. + b**4.)**3.)/(96.*x**2.))

    g += G*Mgalaxy.value*Msun/(r*(r+rgal*kpc))

    return g

def vc(r):
    return np.sqrt(grav_acc(r)*r)

r_inner  = 0.1*rvir*kpc
r_outer  = rvir*kpc
radii    = np.linspace(r_inner,r_outer,100)
vc_outer = np.sqrt(r_outer*grav_acc(r_outer))



################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

class HSE:
    def __init__(self, f_cs_HSE = 2.0, f_cgm=0.1):
        self.f_cs_HSE = f_cs_HSE
        self.f_cgm = f_cgm

        def find_rho_ta_HSE(Mass,r_inner, r_outer):
            rho = lambda r: np.square(vc(r_outer) / vc(r)) * (r/r_outer)**(-gamma*self.f_cs_HSE)
            m_shell = lambda r: 4*np.pi*r**2 * rho(r)
            M_cgm = integrate.quad(m_shell,r_inner, r_outer)[0]
            return Mass/M_cgm

        self.rho_ta_HSE = find_rho_ta_HSE(self.f_cgm*Mhalo, 0.1*rvir*kpc, rvir*kpc)


    def rho(self,r):
        return self.rho_ta_HSE*np.square(vc(rvir*kpc) / vc(r)) * (r/(rvir*kpc))**(-gamma*self.f_cs_HSE)

    def n(self,r):
        return self.rho(r)/mu/mp

    def cs(self,r):
        return vc(r)/np.sqrt(self.f_cs_HSE)

    def T(self,r):
        return mu*mp*self.cs(r)**2/kb/gamma

    def P(self,r):
        return self.n(r) * self.T(r)

    def K(self,r):
        return self.T(r) * self.n(r)**(-2/3.)

    def tcool_P(self,T, P, metallicity):
        return 1.5 * (muH/mu)**2 * kb * T / ( P/T * Lambda((np.log10(P/T*(mu/muH)),np.log10(T), metallicity, redshift)))

    def M_enc(self,r):
        m_shell = lambda radius: 4*np.pi*radius**2 * self.rho(radius)
        M_cgm = integrate.quad(m_shell,0.1*rvir*kpc,r)[0]
        return M_cgm

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

class HSE_turb:
    """
    Mach = dv_turb/cs 
    """
    def __init__(self, f_cs_HSE_turb = 2.0, f_cgm=0.1, Mach=0.5):
        self.f_cs_HSE_turb = f_cs_HSE_turb
        self.f_cgm = f_cgm
        self.Mach = Mach
        def find_rho_ta_HSE_turb(Mass,r_inner, r_outer):
            rho = lambda r: np.square(vc(r_outer) / vc(r)) * (r/r_outer)**(-gamma*self.f_cs_HSE_turb / (1.0 + self.Mach**2))
            m_shell = lambda r: 4*np.pi*r**2 * rho(r)
            M_cgm = integrate.quad(m_shell,r_inner, r_outer)[0]
            return Mass/M_cgm

        self.rho_ta_HSE_turb = find_rho_ta_HSE_turb(self.f_cgm*Mhalo, 0.1*rvir*kpc, rvir*kpc)


    def rho(self,r):
        return self.rho_ta_HSE_turb*np.square(vc(rvir*kpc) / vc(r)) * (r/(rvir*kpc))**(-gamma*self.f_cs_HSE_turb/ (1.0 + self.Mach**2))

    def n(self,r):
        return self.rho(r)/mu/mp

    def cs(self,r):
        return vc(r)/np.sqrt(self.f_cs_HSE_turb)

    def T(self,r):
        return mu*mp*self.cs(r)**2/kb/gamma

    def P(self,r):
        return self.n(r) * self.T(r)

    def K(self,r):
        return self.T(r) * self.n(r)**(-2/3.)

    def tcool_P(self,T, P, metallicity):
        return 1.5 * (muH/mu)**2 * kb * T / ( P/T * Lambda((np.log10(P/T*(mu/muH)),np.log10(T), metallicity, redshift)))

    def M_enc(self,r):
        m_shell = lambda radius: 4*np.pi*radius**2 * self.rho(radius)
        M_cgm = integrate.quad(m_shell,0.1*rvir*kpc,r)[0]
        return M_cgm

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

class HSE_rot:
    def __init__(self, f_cs_HSE_rot = 2.0, f_cgm=0.1, lam=0.05):
        self.f_cs_HSE_rot = f_cs_HSE_rot
        self.f_cgm = f_cgm
        self.lam = lam
        self.r_circ = self.lam*rvir*kpc

        Nr = 1000
        Ntheta=500
        r_edges = np.linspace(r_inner,r_outer,Nr+1)
        theta_edges = np.arccos(np.linspace(-0.99,0.99,Ntheta+1))
        r_centers = r_edges[:-1] + 0.5*np.diff(r_edges)
        thetas = theta_edges[:-1] + 0.5*np.diff(theta_edges)

        gamma = 5/3.
        self.r0 = rvir*kpc
        def rho_outer(r,theta):
            rho = 1
            rho*= (vc(r)/vc(self.r0))**-2
            rho*= (r/self.r0)**(-gamma*self.f_cs_HSE_rot)
            rho*= np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*np.sin(theta)**2 * r**2))
            return rho

        def rho_inner(r,theta):
            rho = (self.r0/self.r_circ)**(gamma*1.0) * np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*self.r_circ**2))
            rho*= (vc(r)/vc(self.r0))**-2
            rho*= (r/self.r0)**(-gamma*(self.f_cs_HSE_rot-1.0))
            rho*= np.sin(theta)**(gamma*1.0)
            return rho


        def rho_HSE_rot(r,theta):
            R = r * np.sin(theta)
            rho = mp * 0.62 
            if R > self.r_circ:
                rho*= (vc(r)/vc(self.r0))**-2
                rho*= (r/self.r0)**(-gamma*self.f_cs_HSE_rot)
                rho*= np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*np.sin(theta)**2 * r**2))
                return rho
            else:
                rho*= (self.r0/self.r_circ)**(gamma*1.0) * np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*self.r_circ**2))
                rho*= (vc(r)/vc(self.r0))**-2
                rho*= (r/self.r0)**(-gamma*(self.f_cs_HSE_rot-1.0))
                rho*= np.sin(theta)**(gamma*1.0)
                return rho
        rho_HSE_rot = np.vectorize(rho_HSE_rot)


        def integral_argument(r,costheta):
            """
            2 pi r^2 rho(r,theta) dr dcostheta
            """
            return 2*np.pi*r**2 * rho_HSE_rot(r,np.arccos(costheta))

        Nr = 100
        Ntheta = 50
        r_edges = np.linspace(0.1*rvir*kpc,rvir*kpc,Nr+1)
        theta_edges = np.arccos(np.linspace(0.99,-0.99,Ntheta+1))
        r_centers = r_edges[:-1] + 0.5*np.diff(r_edges)
        thetas = theta_edges[:-1] + 0.5*np.diff(theta_edges)
        mass = np.sum(np.sum(np.array([integral_argument(r_centers, np.cos(t))*np.diff(r_edges) for t in thetas]),axis=1)*np.diff(theta_edges))
        self.n_ta=(self.f_cgm*10**lMhalo * Msun)/mass
        self.rho_ta = self.n_ta * mu*mp

        # def integral_argument(r,theta):
        #     """
        #     2 pi r^2 rho(r,theta) dr dcostheta
        #     """
        #     return 2*np.pi*r**2 * rho_HSE_rot(r,theta)

        # mass = integrate.dblquad(integral_argument, 0.1*rvir*kpc, 1.0*rvir*kpc, lambda x: -0.99, lambda x: 0.99)
        # self.n_ta=(self.f_cgm*10**lMhalo * Msun)/mass
        # self.rho_ta = self.n_ta * mu*mp

        # Nr = 100
        # Ntheta = 50
        # r_edges = np.linspace(0.1*rvir*kpc,rvir*kpc,Nr+1)
        # theta_edges = np.arccos(np.linspace(0.99,-0.99,Ntheta+1))
        # r_centers = r_edges[:-1] + 0.5*np.diff(r_edges)
        # thetas = theta_edges[:-1] + 0.5*np.diff(theta_edges)
        # mass = np.array([integral_argument(r_centers, t) for t in thetas])
        # self.n_ta=(self.f_cgm*10**lMhalo * Msun)/(np.sum(mass)*np.diff(thetas)[0]*np.diff(r_centers)[0]) 
        # self.rho_ta = self.n_ta * mu*mp

    def rho(self,r,theta):
        R = r * np.sin(theta)
        rho = self.rho_ta
        if R > self.r_circ:
            rho*= (vc(r)/vc(self.r0))**-2
            rho*= (r/self.r0)**(-gamma*self.f_cs_HSE_rot)
            rho*= np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*np.sin(theta)**2 * r**2))
            return rho
        else:
            rho*= (self.r0/self.r_circ)**(gamma*1.0) * np.exp(-gamma*self.f_cs_HSE_rot*self.lam**2*(rvir*kpc)**2 / (2*self.r_circ**2))
            rho*= (vc(r)/vc(self.r0))**-2
            rho*= (r/self.r0)**(-gamma*(self.f_cs_HSE_rot-1.0))
            rho*= np.sin(theta)**(gamma*1.0)
            return rho
    # self.rho = np.vectorize(self.rho)

    def average_rho(self,r):
        return np.array([ np.mean(np.array([self.rho(R,THETA) for THETA in np.arccos(np.linspace(0.99,-0.99,100)) ])) for R in r])

    def n(self,r):
        return self.average_rho(r)/mu/mp

    def cs(self,r):
        return vc(r)/np.sqrt(self.f_cs_HSE_rot)

    def T(self,r):
        return mu*mp*self.cs(r)**2/kb/gamma

    def P(self,r):
        return self.n(r) * self.T(r)

    def K(self,r):
        return self.T(r) * self.n(r)**(-2/3.)

    def tcool_P(self,T, P, metallicity):
        return 1.5 * (muH/mu)**2 * kb * T / ( P/T * Lambda((np.log10(P/T*(mu/muH)),np.log10(T), metallicity, redshift)))

    def v_phi(self,r,theta):
        R = r * np.sin(theta)
        if R > self.r_circ:
            return vc(r) * self.r_circ/R
        else:
            return vc(r)

    def average_v_phi(self,r):
        return np.array([ np.mean(np.array([self.v_phi(R,THETA) for THETA in np.arccos(np.linspace(-0.99,0.99,100))]))for R in r ])


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
"""
Cooling Flow
"""

class cooling_flow:
    def __init__(self, f_cs_CF = 2.0, Mdot = -3.0 * Msun/yr):
        self.f_cs_CF = f_cs_CF
        self.Mdot = Mdot

        TT = lambda r: mu*mp*vc(r)**2 / self.f_cs_CF / gamma /kb
        def dlogrhodlogr(logrho,logr):
            r = np.exp(logr)
            rho = np.exp(logrho)
            vr = self.Mdot/(4*pi*r**2*rho)
            return (self.f_cs_CF - (2/3.)*(r*Lambda((np.log10(rho/muH/mp), np.log10(TT(r)), metallicity, 0.0))*rho) / ( vr * vc(r)**2 * (mu*mp)**2 ) - 2.0) / ( (vr/vc(r))**2 * self.f_cs_CF - 1.0 ) - 2.0

        def rho_coolingflow(r_inner=0.05*rvir*kpc, r_outer=2.0*rvir*kpc, rho_inner = Mhalo/(rvir*kpc)**3):
            logr0 = np.log(r_inner)
            logrmax = np.log(r_outer)
            logrho0 = np.log(rho_inner)

            logrs=[logr0]
            logrhos=[logrho0]

            dlogr = (logrmax - logr0)/1e3

            while logrs[-1] <= logrmax:
                logrhos.append(logrhos[-1]+dlogr*dlogrhodlogr(logrhos[-1],logrs[-1]))
                logrs.append(logrs[-1]+dlogr)
            return np.exp(logrhos), np.exp(logrs)

        self.rho_cf, self.r_cf = rho_coolingflow(r_inner=0.05*rvir*kpc, r_outer=2.0*rvir*kpc, rho_inner = -1.0*self.Mdot/(4*np.pi*(0.02*rvir*kpc)**2*0.99*np.sqrt(kb*TT(0.02*rvir*kpc)/(mu*mp))))#2*Mhalo/(rvir*kpc)**3)


    def T(self,r):
        return mu*mp*vc(r)**2 / self.f_cs_CF / gamma /kb

    def rho(self,r):
        return np.interp(r,self.r_cf,self.rho_cf)

    def n(self,r):
        return np.interp(r,self.r_cf,self.rho_cf)/(mu*mp)

    def vr(self,r):
        return self.Mdot/(4.*pi*r**2*self.rho(r))

    def K(self,r):
        return self.T(r) * (self.n(r))**(-2/3.)

    def P(self,r):
        return self.T(r) * self.n(r)

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
"""
Precipitation
"""

class precipitate:
    def __init__(self, tcooltff=10.0, T_outer=0.25*mu*mp*vc(rvir*kpc)**2/kb):
        self.tcooltff = tcooltff
        self.T_outer = T_outer

        logrs = np.linspace(np.log(0.001*rvir*kpc), np.log(2.1*rvir*kpc),500)
        vc2 = (np.max(vc(np.exp(logrs))))**2
        logTs = np.log(np.logspace(4.2,8,100))

        cooling_slope = interpolate.interp1d(logTs, np.gradient(np.log10(Lambda((-1, np.log10(np.exp(logTs)), metallicity, 0.0))))/np.gradient(logTs) )
        vc_slope = interpolate.interp1d( logrs , np.gradient(np.log10(vc(np.exp(logrs)))) / np.gradient(logrs))

        def dlogTdlogr(logT,logr):
            return (1.0 - vc_slope(logr) - mu * mp * vc(np.exp(logr))**2 / (kb*np.exp(logT)))/(2.0 - cooling_slope(logT))

        def T_precip(r_inner=0.01*rvir*kpc, r_outer=2.0*rvir*kpc, T_outer = self.T_outer):
            logr0 = np.log(r_outer)
            logrmin = np.log(r_inner)
            logT0 = np.log(T_outer)

            logrs=[logr0]
            logTs=[logT0]

            dlogr = -(np.log(r_outer) - np.log(r_inner))/1e3

            while logrs[-1] >= logrmin:
                logTs.append(logTs[-1]+dlogr*dlogTdlogr(logTs[-1],logrs[-1]))
                logrs.append(logrs[-1]+dlogr)
            return np.exp(logTs), np.exp(logrs)

        self.T_precip, self.r_precip = T_precip(r_inner=0.01*rvir*kpc, r_outer=2.0*rvir*kpc, T_outer = self.T_outer)

    def T(self,r):
        return np.interp(r,self.r_precip[::-1], self.T_precip[::-1])

    def rho(self,r):
        return 1.5*mu*mp*kb*self.T(r)*vc(r)/(Lambda((-1,np.log10(self.T(r)),metallicity,0.0))* self.tcooltff * r )

    def n(self,r):
        return self.rho(r)/(mu*mp)

    def K(self,r):
        return self.T(r) * (self.n(r))**(-2/3.)

    def P(self,r):
        return self.T(r) * self.n(r)


################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


line_colors_VW = ['k', "#003366", "#003300", "#3333cc", "#339900", "#66a61e"]


# fn="50_M12subhalos_snap099_TNG100"
# fn="M12_s529643_snap099_TNG100"


files = glob.glob('./data/simulations/*npz')

for fn in files:
    fn = fn[19:-4]
    print fn
    data = np.load('./data/simulations/'+fn+'.npz')

    HSE_halo = HSE(2.0,0.05)                                              #    f_cs_HSE = 2.0, f_cgm=0.1):
    HSE_turb_halo = HSE_turb(2.0,0.05,0.5)                                #    f_cs_HSE_turb = 2.0, f_cgm=0.1, Mach=0.5):
    HSE_rot_halo = HSE_rot(2.0,0.05,0.05)                                 #    f_cs_HSE_rot = 2.0, f_cgm=0.1, lam=0.05):
    cooling_flow_halo = cooling_flow(1.5,-4.0*Msun/yr)                   #    f_cs_CF = 2.0, Mdot = -3.0 * Msun/yr):
    precipitate_halo = precipitate(10.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)   #    tcooltff=10.0, T_outer=0.25*mu*mp*vc(rvir*kpc)**2/kb):




    r_inner  = 0.05*rvir*kpc
    r_outer  = 2.0*rvir*kpc
    radii    = np.linspace(r_inner,r_outer,100)
    vc_outer = np.sqrt(r_outer*grav_acc(r_outer))


    dlogT = np.diff(np.log10(data['temperature_bins']))[0]
    dlogn = np.diff(np.log10(data['number_density_bins']))[0]
    dlogP = np.diff(np.log10(data['pressure_bins']))[0]
    dlogK = np.diff(np.log10(data['entropy_bins']))[0]
    dlogr = np.diff(np.log10(data['r_r200m_profile']))[0]
    dvr = np.diff(data['radial_velocity_bins'])
    dvphi = np.diff(data['azimuthal_velocity_bins'])

    fig,ax = plt.subplots(1,1)

    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Volume']/np.sum(data['temperature_Volume']))/dlogT/dlogr, 
        norm=colors.LogNorm(vmin=1e-2, vmax =3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii),                                color=cm.viridis(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), dashes=[1,2],         color=cm.viridis(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii),     dashes=[4,2],         color=cm.viridis(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.T(radii),      dashes=[4,2,1,2],     color=cm.viridis(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.T(radii),  dashes=[4,2,1,2,1,2], color=cm.viridis(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d \log T \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(3e3,4e7)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    fig,ax = plt.subplots(1,1)

    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii),                                color=cm.plasma(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), dashes=[1,2],         color=cm.plasma(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii),     dashes=[4,2],         color=cm.plasma(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.T(radii),      dashes=[4,2,1,2],     color=cm.plasma(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.T(radii),  dashes=[4,2,1,2,1,2], color=cm.plasma(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d \log T \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(3e3,4e7)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')




    fig,ax = plt.subplots(1,1)

    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Volume']/np.sum(data['number_density_Volume']))/dlogn/dlogr, 
        norm=colors.LogNorm(vmin=1e-2, vmax =3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii),                                color=cm.viridis(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), dashes=[1,2],         color=cm.viridis(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii),     dashes=[4,2],         color=cm.viridis(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.n(radii),      dashes=[4,2,1,2],     color=cm.viridis(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.n(radii),  dashes=[4,2,1,2,1,2], color=cm.viridis(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d \log n \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(5e-6,3e0)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()



    fig,ax = plt.subplots(1,1)

    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii),                                color=cm.plasma(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), dashes=[1,2],         color=cm.plasma(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii),     dashes=[4,2],         color=cm.plasma(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.n(radii),      dashes=[4,2,1,2],     color=cm.plasma(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.n(radii),  dashes=[4,2,1,2,1,2], color=cm.plasma(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d \log n \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(5e-6,3e0)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')




    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Volume']/np.sum(data['pressure_Volume']))/dlogP/dlogr, 
        norm=colors.LogNorm(vmin=1e-2, vmax =3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii),                                color=cm.viridis(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.P(radii), dashes=[1,2],         color=cm.viridis(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii),     dashes=[4,2],         color=cm.viridis(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.P(radii),      dashes=[4,2,1,2],     color=cm.viridis(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.P(radii),  dashes=[4,2,1,2,1,2], color=cm.viridis(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d \log P \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(1e-1,1e6)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Mass']/(fb*Mhalo/Msun)/dlogP/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii),                                color=cm.plasma(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.P(radii), dashes=[1,2],         color=cm.plasma(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii),     dashes=[4,2],         color=cm.plasma(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.P(radii),      dashes=[4,2,1,2],     color=cm.plasma(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.P(radii),  dashes=[4,2,1,2,1,2], color=cm.plasma(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d \log P \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    ax.set_ylim(1e-1,1e6)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=2,columnspacing=-3, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Volume']/np.sum(data['entropy_Volume']))/dlogK/dlogr, 
        norm=colors.LogNorm(vmin=1e-2, vmax =3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii),                                color=cm.viridis(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.K(radii), dashes=[1,2],         color=cm.viridis(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii),     dashes=[4,2],         color=cm.viridis(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.K(radii),      dashes=[4,2,1,2],     color=cm.viridis(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.K(radii),  dashes=[4,2,1,2,1,2], color=cm.viridis(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d \log K \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Mass']/(fb*Mhalo/Msun)/dlogK/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii),                                color=cm.plasma(0.0), lw=2.5, label='HSE')
    ax.plot(radii/(rvir*kpc), cooling_flow_halo.K(radii), dashes=[1,2],         color=cm.plasma(0.2), lw=2.5, label='Cooling Flow'),
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii),     dashes=[4,2],         color=cm.plasma(0.4), lw=2.5, label='HSE w/'+r'$\mathcal{M}=0.5$'+' turb.')
    ax.plot(radii/(rvir*kpc), HSE_rot_halo.K(radii),      dashes=[4,2,1,2],     color=cm.plasma(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    ax.plot(radii/(rvir*kpc), precipitate_halo.K(radii),  dashes=[4,2,1,2,1,2], color=cm.plasma(0.8), lw=2.5, label='HSE w/ ' + r'$\frac{t_{\rm cool}}{t_{\rm ff}} = 10$'+' precip.')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d \log K \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')




    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (((data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume']))).T/dvr).T /dlogr, 
        norm=colors.LogNorm(vmin=1e-3,vmax=3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, dashes=[1,2],         color=cm.viridis(0.2), lw=2.5, label='Cooling Flow'),
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d v_r \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (data['radial_velocity_Mass'].T/(fb*Mhalo/Msun)/dvr /dlogr).T, 
        norm=colors.LogNorm(vmin=3e-4, vmax=3e-1), cmap='viridis')

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, dashes=[1,2],         color=cm.plasma(0.2), lw=2.5, label='Cooling Flow'),
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d v_r \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (((data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume']))).T/dvphi).T /dlogr, 
        norm=colors.LogNorm(vmin=1e-3,vmax=3), cmap='plasma')

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5,  dashes=[4,2,1,2],     color=cm.viridis(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (V/V_{\rm tot}) / d v_\phi \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (data['azimuthal_velocity_Mass'].T/(fb*Mhalo/Msun)/dvphi /dlogr).T, 
        norm=colors.LogNorm(vmin=3e-4, vmax=3e-1), cmap='viridis')

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5,  dashes=[4,2,1,2],     color=cm.plasma(0.6), lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$')
    cb = fig.colorbar(plot)
    cb.set_label(r'$d^2 (M/ f_b M_{\rm halo}) / d v_\phi \, d \log r$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ax.set_xlim(5e-2, 2)
    # ax.set_ylim(1e-1,1e6)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1, handlelength=3.0)
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')
















    ir = 5

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        data['pressure_entropy_Volume'][...,ir]/np.sum(data['pressure_entropy_Volume'][...,ir]), 
        norm=colors.LogNorm(vmin=1e-5,vmax=1e-1), cmap='plasma')
    cb = fig.colorbar(plot)
    cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$',fontsize=10)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_entropy_Volume_r_'+str(ir).zfill(3)+'_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir]), 
        norm=colors.LogNorm(vmin=1e-5,vmax=1e-1), cmap='viridis')
    cb = fig.colorbar(plot)
    cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$',fontsize=10)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_entropy_Mass_r_'+str(ir).zfill(3)+'_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       










    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        np.sum(data['pressure_entropy_Volume'],axis=-1)/np.sum(data['pressure_entropy_Volume']), 
        norm=colors.LogNorm(vmin=1e-5,vmax=1e-1), cmap='plasma')
    cb = fig.colorbar(plot)
    cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_entropy_Volume_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       



    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        np.sum(data['pressure_entropy_Mass'],axis=-1)/(fb*Mhalo/Msun)/dlogK/dlogP,
        norm=colors.LogNorm(vmin=1e-3,vmax=1e-0), cmap='binary')
    cb = fig.colorbar(plot)
    cb.set_label(r'$\frac{d^2 \log (M/ f_b M_{\rm halo})}{d \log P \, d \log K}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    ir = 1
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP),
        [1.5e-2], colors='#6600cc',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#6600cc", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP),
        [1.5e-2], colors='#0066ff',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#0066ff", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP),
        [1.5e-2], colors='#66cc00',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#66cc00", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP),
        [1.5e-2], colors='#ff9900',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#ff9900", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP),
        [1.5e-2], colors='#cc3300',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#cc3300", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_entropy_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       



    from matplotlib import gridspec

    fig = plt.figure()
    gs  = gridspec.GridSpec(100, 100, wspace=0.1, hspace=0.05)
    cax = plt.subplot(gs[15:85, 88:  ])
    ax  = plt.subplot(gs[:,      0:82])
    for ir in np.arange(0,11,1):
        # ir = 19-ir
        color = cm.Spectral(ir/10.)
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] *= color[0]
        vals[:, 1] *= color[1]
        vals[:, 2] *= color[2]
        vals[:, 3] *= np.linspace(1.0,0,N)
        newcmp = ListedColormap(vals[::-1])

        logmassfrac=np.log10(data['pressure_entropy_Mass'][...,ir]/(fb*Mhalo/Msun)/dlogK/dlogP)
        logmassfrac[np.isinf(logmassfrac)] = -10.

        plot=ax.contourf(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
            logmassfrac, np.linspace(-3,-0.9,100),
            vmin=-3,vmax=-1, cmap=newcmp, antialiased=True, extend='both')

        cax.contourf(data['r_r200m_phase'][ir:ir+2]-0.05, np.linspace(-3,-0.9,100), np.array([np.linspace(-3,-0.9,100),np.linspace(-3,-0.9,100)]).T,50,cmap=newcmp, antialiased=True)

    cax.yaxis.set_label_position("right")
    cax.yaxis.tick_right()
    cax.set_title(r'$\frac{d^2 \log (M/ f_b M_{\rm halo})}{d \log P \, d \log K}$')#, fontsize=10,rotation=270,labelpad=15)
    cax.set_xlabel(r'$r/r_{\rm 200m}$',fontsize=10)
    # cb.set_ticks(np.arange(-6,-2.9,1))
    # cb.ax.minorticks_off()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_title(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_entropy_Mass_afew_r_rainbow_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       




























    HSE_halo = HSE(2.0,0.05)                                              #    f_cs_HSE = 2.0, f_cgm=0.1):
    HSE_turb_halo = HSE_turb(2.0,0.05,0.5)                                #    f_cs_HSE_turb = 2.0, f_cgm=0.1, Mach=0.5):
    HSE_rot_halo = HSE_rot(2.0,0.05,0.05)                                 #    f_cs_HSE_rot = 2.0, f_cgm=0.1, lam=0.05):
    cooling_flow_halo = cooling_flow(1.5,-4.0*Msun/yr)                   #    f_cs_CF = 2.0, Mdot = -3.0 * Msun/yr):
    precipitate_halo = precipitate(10.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)   #    tcooltff=10.0, T_outer=0.25*mu*mp*vc(rvir*kpc)**2/kb):







    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_halo = HSE(4.0,0.05)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 4$')
    HSE_halo = HSE(2.0,0.05)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 2$')
    HSE_halo = HSE(1.0,0.05)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 1$')
    HSE_halo = HSE(0.5,0.05)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 1/2$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_halo = HSE(4.0,0.05)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 4$')
    HSE_halo = HSE(2.0,0.05)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 2$')
    HSE_halo = HSE(1.0,0.05)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 1$')
    HSE_halo = HSE(0.5,0.05)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{v_c^2}{c_s^2} = 1/2$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/HSE_f_cs_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()



    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_halo = HSE(2.0,0.01)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.01$')
    HSE_halo = HSE(2.0,0.05)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.05$')
    HSE_halo = HSE(2.0,0.1)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.1$')
    HSE_halo = HSE(2.0,0.15)                                              
    axarr[0].plot(radii/(rvir*kpc), HSE_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.15$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_halo = HSE(2.0,0.01)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.01$')
    HSE_halo = HSE(2.0,0.05)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.05$')
    HSE_halo = HSE(2.0,0.1)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.1$')
    HSE_halo = HSE(2.0,0.15)                                              
    axarr[1].plot(radii/(rvir*kpc), HSE_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{M_{\rm CGM}}{M_{\rm halo}} = 0.15$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/HSE_f_cgm_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()














    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_rot_halo = HSE_rot(2.0,0.05,0.01)
    axarr[0].plot(radii/(rvir*kpc), HSE_rot_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\lambda = 0.01$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.05)
    axarr[0].plot(radii/(rvir*kpc), HSE_rot_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\lambda = 0.05$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.1)
    axarr[0].plot(radii/(rvir*kpc), HSE_rot_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\lambda = 0.1$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.15)
    axarr[0].plot(radii/(rvir*kpc), HSE_rot_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\lambda = 0.15$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    HSE_rot_halo = HSE_rot(2.0,0.05,0.01)
    axarr[1].plot(radii/(rvir*kpc), HSE_rot_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\lambda = 0.01$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.05)
    axarr[1].plot(radii/(rvir*kpc), HSE_rot_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\lambda = 0.05$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.1)
    axarr[1].plot(radii/(rvir*kpc), HSE_rot_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\lambda = 0.1$')
    HSE_rot_halo = HSE_rot(2.0,0.05,0.15)
    axarr[1].plot(radii/(rvir*kpc), HSE_rot_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\lambda = 0.15$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/HSE_rot_lambda_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

















    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    cooling_flow_halo = cooling_flow(1.5,-0.5*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\dot{M} = 0.5 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-1.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\dot{M} = 1 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-2.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\dot{M} = 2 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-4.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\dot{M} = 4 M_\odot/{\rm yr}$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    cooling_flow_halo = cooling_flow(1.5,-0.5*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\dot{M} = 0.5 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-1.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\dot{M} = 1.0 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-2.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\dot{M} = 2 M_\odot/{\rm yr}$')
    cooling_flow_halo = cooling_flow(1.5,-4.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\dot{M} = 4 M_\odot/{\rm yr}$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/cooling_flow_fcs15_mdot_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()




    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    cooling_flow_halo = cooling_flow(4.0,-2.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=4.0$')
    cooling_flow_halo = cooling_flow(2.0,-2.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=2.0$')
    cooling_flow_halo = cooling_flow(1.0,-2.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=1.0$')
    cooling_flow_halo = cooling_flow(0.5,-2.0*Msun/yr)
    axarr[0].plot(radii/(rvir*kpc), cooling_flow_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=0.5$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    cooling_flow_halo = cooling_flow(4.0,-2.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=4.0$')
    cooling_flow_halo = cooling_flow(2.0,-2.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=2.0$')
    cooling_flow_halo = cooling_flow(1.0,-2.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=1.0$')
    cooling_flow_halo = cooling_flow(0.5,-2.0*Msun/yr)
    axarr[1].plot(radii/(rvir*kpc), cooling_flow_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{v_c^2}{c_s^2}=0.5$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/cooling_flow_mdot_2_fcs_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()























    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(1.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=1$')
    precipitate_halo = precipitate(3.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=3$')
    precipitate_halo = precipitate(10.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=10$')
    precipitate_halo = precipitate(30.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=30$')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(1.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.0), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=1$')
    precipitate_halo = precipitate(3.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=3$')
    precipitate_halo = precipitate(10.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=10$')
    precipitate_halo = precipitate(30.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$\frac{t_{\rm cool}}{t_{\rm ff}}=30$')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/precipitate_tcool_tff_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()




    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(10.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$T_{\rm out}=\frac{1}{20} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(10.0,0.2*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$T_{\rm out}=\frac{1}{4} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(10.0,1.0*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$T_{\rm out}= \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(10.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$T_{\rm out}=\frac{1}{20} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(10.0 ,0.2*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$T_{\rm out}=\frac{1}{4} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(10.0 ,1.0*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$T_{\rm out}= \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/precipitate_T_out_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()







    fig, axarr = plt.subplots(2,1,sharex=True)

    plot=axarr[0].pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/(fb*Mhalo/Msun)/dlogT/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(3.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.3), lw=2.5, label=r'$T_{\rm out}=\frac{1}{20} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(3.0,0.2*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.6), lw=2.5, label=r'$T_{\rm out}=\frac{1}{4} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(3.0,1.0*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[0].plot(radii/(rvir*kpc), precipitate_halo.T(radii), color=cm.plasma(0.9), lw=2.5, label=r'$T_{\rm out}= \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')

    cb = fig.colorbar(plot,ax=axarr[0])
    cb.set_label(r'$\frac{d^2 (M/f_b M_{\rm halo}) }{d \log T \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()
    axarr[0].set_xlim(5e-2, 2)
    axarr[0].set_ylim(3e3,4e7)
    axarr[0].set_yscale('log')
    axarr[0].set_xscale('log')
    axarr[0].set_ylabel(r'$T\,[\mathrm{K}]$')
    axarr[0].legend(loc='upper left', fontsize=6,ncol=4, columnspacing=1)
    axarr[0].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)



    plot=axarr[1].pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/(fb*Mhalo/Msun)/dlogn/dlogr), 
        norm=colors.LogNorm(vmin=1e-2, vmax=3), cmap='viridis')

    precipitate_halo = precipitate(3.0 ,0.05*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.3), lw=2.5, label=r'$T_{\rm out}=\frac{1}{20} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(3.0 ,0.2*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.6), lw=2.5, label=r'$T_{\rm out}=\frac{1}{4} \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')
    precipitate_halo = precipitate(3.0 ,1.0*mu*mp*vc(rvir*kpc)**2/kb)
    axarr[1].plot(radii/(rvir*kpc), precipitate_halo.n(radii), color=cm.plasma(0.9), lw=2.5, label=r'$T_{\rm out}= \frac{\mu m_p}{k_B} v_{\rm vir}^2 $')

    cb = fig.colorbar(plot,ax=axarr[1])
    cb.set_label(r'$\frac{d^2 (M/ f_b M_{\rm halo})}{ d \log n \, d \log r}$',rotation=270,fontsize=12,labelpad=15)
    cb.ax.minorticks_off()

    axarr[1].set_xlim(5e-2, 2)
    axarr[1].set_ylim(5e-6,1e-1)
    axarr[1].set_yscale('log')
    axarr[1].set_xscale('log')
    axarr[1].set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    axarr[1].set_xlabel(r'$r/r_{\rm vir}$')
    axarr[1].legend(loc='upper right', fontsize=6,ncol=2, columnspacing=1)
    axarr[1].grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)

    # fig.suptitle(r'$t='+str(np.round(data['time'],2))+r'\,\mathrm{Gyr}$', y = 0.96)
    fig.set_size_inches(5,5)
    plt.savefig('./plots/precipitate_tcooltff3_T_out_comparison_T_n_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
