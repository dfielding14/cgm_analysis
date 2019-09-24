import glob
import numpy as np
import h5py
from scipy import integrate, interpolate
from astropy import units as un, constants as cons
from astropy.cosmology import Planck15 as cosmo
import colossus, colossus.cosmology.cosmology
colossus.cosmology.cosmology.setCosmology('planck15')
from colossus.halo import profile_dk14
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
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import palettable
from scipy.ndimage.filters import gaussian_filter

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
    for i in range(len(files)):
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





def c_DuttonMaccio14(lMhalo, z=0):  #table 3 appropriate for Mvir
    c_z0  = lambda lMhalo: 10.**(1.025 - 0.097*(lMhalo-np.log10(0.7**-1*1e12))) 
    c_z05 = lambda lMhalo: 10.**(0.884 - 0.085*(lMhalo-np.log10(0.7**-1*1e12))) 
    c_z1  = lambda lMhalo: 10.**(0.775 - 0.073*(lMhalo-np.log10(0.7**-1*1e12))) 
    c_z2  = lambda lMhalo: 10.**(0.643 - 0.051*(lMhalo-np.log10(0.7**-1*1e12)))
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
    sms = zparams['sm_0'] - np.log10(10**(-zparams['alpha']*dms) + 10**(-zparams['beta']*dms)) + zparams['gamma']*np.e**(-0.5*(dm2s*dm2s))
    return ms,sms

def MgalaxyBehroozi(lMhalo, z, parameter_file='./data/smhm_true_med_cen_params.txt'):
    ms,sms = Behroozi_params(z,parameter_file)
    lMstar = interpolate.interp1d(ms, sms, fill_value='extrapolate')(lMhalo)
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



#### Fit to Diemer+14 nu vs logMhalo ---- THIS ONLY works for z = 0
nus = np.array([0.720000,0.7777,0.845000,1.200000,1.845400,3.140000])
logMhalos = np.array([11.5,11.75,12.,13.,14.,15.])

lMhalo=12.
nu = np.interp(lMhalo, logMhalos, nus)


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

class HSE_adiabat:
    def __init__(self, f_Kvir = 1.0, f_cgm=0.1):
        self.f_Kvir = f_Kvir
        self.f_cgm = f_cgm

        self.K0 = self.f_Kvir * 0.5*vc(rvir*kpc)**2 * ((fb * Mhalo / (4 * np.pi/3. * (rvir*kpc)**3))**(-2./3.))

        def find_rho_ta_HSE_adiabat(Mass,r_inner, r_outer):
            rho = lambda r, rho_0: (rho_0**(2/3.) + (gamma-1)/(gamma* self.K0) * scipy.integrate.quad(grav_acc, r, rvir*kpc)[0])**(3/2.)
            m_shell = lambda r, rho_0: 4*np.pi*r**2 * rho(r, rho_0)
            rho_0s = np.logspace(-33,-23)
            M_cgms = np.zeros_like(rho_0s)
            for i in range(len(rho_0s)):
                M_cgms[i] = integrate.quad(m_shell,r_inner, r_outer, args=(rho_0s[i]))[0]

            return np.interp(Mass, M_cgms, rho_0s)

        self.rho_ta_HSE_adiabat = find_rho_ta_HSE_adiabat(self.f_cgm*Mhalo, 0.1*rvir*kpc, rvir*kpc)

        self.rho = np.vectorize(self.rho_unvec)


    def rho_unvec(self,r):
        return (self.rho_ta_HSE_adiabat**(2/3.) + (gamma-1)/(gamma*self.K0) * scipy.integrate.quad(grav_acc, r, rvir*kpc)[0])**(3/2.)   

    def n(self,r):
        return self.rho(r)/mu/mp

    def K(self,r):
        return self.K0 * np.ones_like(r)/kb * (mu*mp)**(gamma)

    def cs(self,r):
        return np.sqrt(gamma * self.K0 * self.rho(r)**(gamma-1))

    def T(self,r):
        return mu*mp*self.cs(r)**2/kb/gamma

    def P(self,r):
        return self.n(r) * self.T(r)

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
            vr = self.Mdot/(4*np.pi*r**2*rho)
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
        return self.Mdot/(4.*np.pi*r**2*self.rho(r))

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

HSE_halo            = HSE(2.0,0.05)                                     #    f_cs_HSE = 2.0, f_cgm=0.1):
HSE_adiabat_halo    = HSE_adiabat(1.0,0.05)                             #    f_Kvir = 1.0, f_cgm=0.1):
HSE_turb_halo       = HSE_turb(2.0,0.05,0.5)                            #    f_cs_HSE_turb = 2.0, f_cgm=0.1, Mach=0.5):
HSE_rot_halo        = HSE_rot(2.0,0.05,0.05)                            #    f_cs_HSE_rot = 2.0, f_cgm=0.1, lam=0.05):
cooling_flow_halo   = cooling_flow(1.5,-4.0*Msun/yr)                    #    f_cs_CF = 2.0, Mdot = -3.0 * Msun/yr):
precipitate_halo    = precipitate(10.0,0.05*mu*mp*vc(rvir*kpc)**2/kb)   #    tcooltff=10.0, T_outer=0.25*mu*mp*vc(rvir*kpc)**2/kb):

# palette_Volume = palettable.cubehelix.Cubehelix.make(start=1.5, rotation=2/3., reverse=True, max_light=1.0, min_light=0.1, n=64)
# palette_Mass   = palettable.cubehelix.Cubehelix.make(start=2.5, rotation=-2/3., reverse=True, max_light=1.0, min_light=0.1, n=64)

def plotter(data,fn, palet):
    dlogT = np.diff(np.log10(data['temperature_bins']))[0]
    dlogn = np.diff(np.log10(data['number_density_bins']))[0]
    dlogP = np.diff(np.log10(data['pressure_bins']))[0]
    dlogK = np.diff(np.log10(data['entropy_bins']))[0]
    dlogr = np.diff(np.log10(data['r_r200m_profile']))[0]
    dvr = np.diff(data['radial_velocity_bins'])
    dvphi = np.diff(data['azimuthal_velocity_bins'])


        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(T,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/np.sum(data['temperature_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(T,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(n,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/np.sum(data['number_density_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(n,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(P,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Mass']/np.sum(data['pressure_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(P,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')

        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(K,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Mass']/np.sum(data['entropy_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(K,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')

        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, lw=2.5, label='cooling flow', color = '#cc3300', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(v_r,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper left', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (data['radial_velocity_Mass']/np.sum(data['radial_velocity_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, lw=2.5, label='cooling flow', color = '#0066ff', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper left', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


        ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5, lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$', color = '#cc3300', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(v_\phi,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper left', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (data['azimuthal_velocity_Mass']/np.sum(data['azimuthal_velocity_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5, lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$', color = '#0066ff', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper left', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################


    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        np.sum(data['pressure_entropy_Mass'][...,2:4],axis=-1)/np.sum(data['pressure_entropy_Mass'][...,2:4]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(P,K,0.2\leq r/r_{\rm vir} \leq 0.4) / M(0.2\leq r/r_{\rm vir} \leq 0.4)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))

    axR.plot(np.sum(data['pressure_entropy_Mass'][...,2:4],axis=(1,2))/np.sum(data['pressure_entropy_Mass'][...,2:4])/dlogK,data['entropy_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    plt.savefig('./plots/pressure_entropy_Mass_02r04_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

        ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        np.sum(data['pressure_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['pressure_entropy_Mass'][...,1:10]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(P,K,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))

    ir = 1
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir])/dlogK,data['entropy_bins'][:-1], color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir])/dlogK,data['entropy_bins'][:-1], color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir])/dlogK,data['entropy_bins'][:-1], color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir])/dlogK,data['entropy_bins'][:-1], color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        (data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir])/dlogK,data['entropy_bins'][:-1], color='0.8', lw=2.)

    ax.legend(loc='lower left', fontsize=6, ncol=1)


    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    plt.savefig('./plots/pressure_entropy_Mass_all_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['number_density_bins'], data['temperature_bins'], 
        np.sum(data['density_temperature_Mass'][...,2:4],axis=-1)/np.sum(data['density_temperature_Mass'][...,2:4]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(n,T,0.2\leq r/r_{\rm vir} \leq 0.4) / M(0.2\leq r/r_{\rm vir} \leq 0.4)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim((3e3,1e7))
    ax.set_xlim((3e-6,0.8e-1))

    axR.plot(np.sum(data['density_temperature_Mass'][...,2:4],axis=(1,2))/np.sum(data['density_temperature_Mass'][...,2:4])/dlogK,data['temperature_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((3e3,1e7))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log T}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    plt.savefig('./plots/density_temperature_Mass_02r04_'+fn+'_with_dMdlogT.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['number_density_bins'], data['temperature_bins'], 
        np.sum(data['density_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['density_temperature_Mass'][...,1:10]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(n,T,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim((3e3,1e7))
    ax.set_xlim((3e-6,0.8e-1))

    ir = 1
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        (data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir])/dlogK,data['temperature_bins'][:-1], color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        (data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir])/dlogK,data['temperature_bins'][:-1], color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        (data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir])/dlogK,data['temperature_bins'][:-1], color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        (data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir])/dlogK,data['temperature_bins'][:-1], color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        (data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir])/dlogK,data['temperature_bins'][:-1], color='0.8', lw=2.)

    ax.legend(loc='lower left', fontsize=6, ncol=1)


    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((3e3,1e7))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log T}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    plt.savefig('./plots/density_temperature_Mass_all_'+fn+'_with_dMdlogT.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['azimuthal_velocity_bins'], data['entropy_bins'], 
        np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10]),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,K)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.0", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.2", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.4", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.6", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.8", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e4,1e10))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_entropy_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       

    ##############################################################################################################


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['azimuthal_velocity_bins'], data['temperature_bins'], 
        np.sum(data['azimuthal_velocity_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_temperature_Mass'][...,1:10]),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,T)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='#6600cc',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#6600cc", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='#0066ff',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#0066ff", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='#66cc00',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#66cc00", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='#ff9900',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#ff9900", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='#cc3300',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#cc3300", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e3,3e7))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_temperature_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['radial_velocity_bins'], data['temperature_bins'], 
        np.sum(data['radial_velocity_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_temperature_Mass'][...,1:10]),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,T)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        (data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower right', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e3,3e7))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_temperature_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        np.sum(data['radial_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,1:10]),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,K)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower right', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e4,3e10))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_entropy_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(80, 65, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[20:80, :60])
    axT= plt.subplot( gs[0:20, :60])
    cax = plt.subplot(gs[20:80, 63:])

    plot = ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        np.sum(data['radial_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,1:10]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='vertical', cax=cax)
    cb.set_label(r'$M(K,v_r,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, rotation=270, labelpad=15)
    # cb.ax.minorticks_off()
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,3e10))
    ax.set_xlim((-300,600))

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir])/np.diff(data['radial_velocity_bins']), color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir])/np.diff(data['radial_velocity_bins']), color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir])/np.diff(data['radial_velocity_bins']), color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir])/np.diff(data['radial_velocity_bins']), color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir])/np.diff(data['radial_velocity_bins']), color='0.8', lw=2.)

    ax.legend(loc='lower right', fontsize=6, ncol=1)


    axT.set_yscale('log')
    axT.set_xticklabels([])

    axT.set_ylim((1e-5,1e-1))
    axT.set_xlim((-300,600))
    axT.yaxis.set_label_position("right")
    axT.set_ylabel(r'$\frac{d \log M}{ d v_r}$', fontsize=14, rotation=270, labelpad=30)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(4,5)
    plt.savefig('./plots/radial_velocity_entropy_Mass_all_'+fn+'_with_dMdvr.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(80, 65, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[20:80, :60])
    axT= plt.subplot( gs[0:20, :60])
    cax = plt.subplot(gs[20:80, 63:])

    plot = ax.pcolormesh(data['azimuthal_velocity_bins'], data['entropy_bins'], 
        np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10]),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='vertical', cax=cax)
    cb.set_label(r'$M(K,v_\phi,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, rotation=270, labelpad=15)
    # cb.ax.minorticks_off()
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,3e10))
    ax.set_xlim((-300,300))

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        (data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.8', lw=2.)

    ax.legend(loc='lower right', fontsize=6, ncol=1)


    axT.set_yscale('log')
    axT.set_xticklabels([])

    axT.set_ylim((1e-5,1e-1))
    axT.set_xlim((-300,300))
    axT.yaxis.set_label_position("right")
    axT.set_ylabel(r'$\frac{d \log M}{ d v_\phi}$', fontsize=14, rotation=270, labelpad=30)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(4,5)
    plt.savefig('./plots/azimuthal_velocity_entropy_Mass_all_'+fn+'_with_dMdvr.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    plt.close("all")

def plotter_linear(data,fn, palet):
    dlogT = np.diff(np.log10(data['temperature_bins']))[0]
    dlogn = np.diff(np.log10(data['number_density_bins']))[0]
    dlogP = np.diff(np.log10(data['pressure_bins']))[0]
    dlogK = np.diff(np.log10(data['entropy_bins']))[0]
    dlogr = np.diff(np.log10(data['r_r200m_profile']))[0]
    dvr = np.diff(data['radial_velocity_bins'])
    dvphi = np.diff(data['azimuthal_velocity_bins'])


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(T,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
        (data['temperature_Mass']/np.sum(data['temperature_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.T(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(T,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/temperature_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(n,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
        (data['number_density_Mass']/np.sum(data['number_density_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.n(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(n,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/number_density_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(P,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
        (data['pressure_Mass']/np.sum(data['pressure_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.P(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(P,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/pressure_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')

    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii), lw=2.5, label='HSE', color = '#cc3300', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#cc3300', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(K,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
        (data['entropy_Mass']/np.sum(data['entropy_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii), lw=2.5, label='HSE', color = '#0066ff', alpha=0.75)
    ax.plot(radii/(rvir*kpc), HSE_turb_halo.K(radii), lw=2.5, label=r'HSE w/ $\mathcal{M} = 0.5$', color = '#0066ff', alpha=0.75, ls='--')
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(K,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/entropy_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')

    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, lw=2.5, label='cooling flow', color = '#cc3300', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(v_r,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
        (data['radial_velocity_Mass']/np.sum(data['radial_velocity_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5, lw=2.5, label='cooling flow', color = '#0066ff', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/radial_velocity_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5, lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$', color = '#cc3300', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'${\rm Vol}(v_\phi,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
        (data['azimuthal_velocity_Mass']/np.sum(data['azimuthal_velocity_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_rot_halo.average_v_phi(radii)/1e5, lw=2.5, label='HSE w/ rot. '+r'$\lambda=0.05$', color = '#0066ff', alpha=0.75)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    plt.savefig('./plots/azimuthal_velocity_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plt.close('all')

def averager(files):
    for i,fn in enumerate(files):
        fn = fn[19:-4]
        print(fn)
        data = np.load('./data/simulations/'+fn+'.npz')
        if i == 0 :
            azimuthal_velocity_Mass               = data['azimuthal_velocity_Mass']
            azimuthal_velocity_entropy_Mass       = data['azimuthal_velocity_entropy_Mass']
            azimuthal_velocity_temperature_Mass   = data['azimuthal_velocity_temperature_Mass']
            density_temperature_Mass              = data['density_temperature_Mass']
            entropy_Mass                          = data['entropy_Mass']
            number_density_Mass                   = data['number_density_Mass']
            polar_velocity_Mass                   = data['polar_velocity_Mass']
            pressure_Mass                         = data['pressure_Mass']
            pressure_entropy_Mass                 = data['pressure_entropy_Mass']
            radial_velocity_Mass                  = data['radial_velocity_Mass']
            radial_velocity_entropy_Mass          = data['radial_velocity_entropy_Mass']
            radial_velocity_temperature_Mass      = data['radial_velocity_temperature_Mass']
            specific_angular_momentum_x_Mass      = data['specific_angular_momentum_x_Mass']
            specific_angular_momentum_y_Mass      = data['specific_angular_momentum_y_Mass']
            specific_angular_momentum_z_Mass      = data['specific_angular_momentum_z_Mass']
            # tcool_Mass                            = data['tcool_Mass']
            temperature_Mass                      = data['temperature_Mass']
            azimuthal_velocity_Volume             = data['azimuthal_velocity_Volume']
            azimuthal_velocity_entropy_Volume     = data['azimuthal_velocity_entropy_Volume']
            azimuthal_velocity_temperature_Volume = data['azimuthal_velocity_temperature_Volume']
            density_temperature_Volume            = data['density_temperature_Volume']
            entropy_Volume                        = data['entropy_Volume']
            number_density_Volume                 = data['number_density_Volume']
            polar_velocity_Volume                 = data['polar_velocity_Volume']
            pressure_Volume                       = data['pressure_Volume']
            pressure_entropy_Volume               = data['pressure_entropy_Volume']
            radial_velocity_Volume                = data['radial_velocity_Volume']
            radial_velocity_entropy_Volume        = data['radial_velocity_entropy_Volume']
            radial_velocity_temperature_Volume    = data['radial_velocity_temperature_Volume']
            specific_angular_momentum_x_Volume    = data['specific_angular_momentum_x_Volume']
            specific_angular_momentum_y_Volume    = data['specific_angular_momentum_y_Volume']
            specific_angular_momentum_z_Volume    = data['specific_angular_momentum_z_Volume']
            # tcool_Volume                          = data['tcool_Volume']
            temperature_Volume                    = data['temperature_Volume']

        azimuthal_velocity_Mass               += data['azimuthal_velocity_Mass']
        azimuthal_velocity_entropy_Mass       += data['azimuthal_velocity_entropy_Mass']
        azimuthal_velocity_temperature_Mass   += data['azimuthal_velocity_temperature_Mass']
        density_temperature_Mass              += data['density_temperature_Mass']
        entropy_Mass                          += data['entropy_Mass']
        number_density_Mass                   += data['number_density_Mass']
        polar_velocity_Mass                   += data['polar_velocity_Mass']
        pressure_Mass                         += data['pressure_Mass']
        pressure_entropy_Mass                 += data['pressure_entropy_Mass']
        radial_velocity_Mass                  += data['radial_velocity_Mass']
        radial_velocity_entropy_Mass          += data['radial_velocity_entropy_Mass']
        radial_velocity_temperature_Mass      += data['radial_velocity_temperature_Mass']
        specific_angular_momentum_x_Mass      += data['specific_angular_momentum_x_Mass']
        specific_angular_momentum_y_Mass      += data['specific_angular_momentum_y_Mass']
        specific_angular_momentum_z_Mass      += data['specific_angular_momentum_z_Mass']
        # tcool_Mass                            += data['tcool_Mass']
        temperature_Mass                      += data['temperature_Mass']
        azimuthal_velocity_Volume             += data['azimuthal_velocity_Volume']
        azimuthal_velocity_entropy_Volume     += data['azimuthal_velocity_entropy_Volume']
        azimuthal_velocity_temperature_Volume += data['azimuthal_velocity_temperature_Volume']
        density_temperature_Volume            += data['density_temperature_Volume']
        entropy_Volume                        += data['entropy_Volume']
        number_density_Volume                 += data['number_density_Volume']
        polar_velocity_Volume                 += data['polar_velocity_Volume']
        pressure_Volume                       += data['pressure_Volume']
        pressure_entropy_Volume               += data['pressure_entropy_Volume']
        radial_velocity_Volume                += data['radial_velocity_Volume']
        radial_velocity_entropy_Volume        += data['radial_velocity_entropy_Volume']
        radial_velocity_temperature_Volume    += data['radial_velocity_temperature_Volume']
        specific_angular_momentum_x_Volume    += data['specific_angular_momentum_x_Volume']
        specific_angular_momentum_y_Volume    += data['specific_angular_momentum_y_Volume']
        specific_angular_momentum_z_Volume    += data['specific_angular_momentum_z_Volume']
        # tcool_Volume                          += data['tcool_Volume']
        temperature_Volume                    += data['temperature_Volume']

        r_r200m_phase                      = data['r_r200m_phase']
        r_r200m_profile                    = data['r_r200m_profile']
        halo_mass                          = data['halo_mass']
        time                               = data['time']
        r200m                              = data['r200m']
        azimuthal_velocity_bins            = data['azimuthal_velocity_bins']
        entropy_bins                       = data['entropy_bins']
        number_density_bins                = data['number_density_bins']
        polar_velocity_bins                = data['polar_velocity_bins']
        pressure_bins                      = data['pressure_bins']
        radial_velocity_bins               = data['radial_velocity_bins']
        specific_angular_momentum_x_bins   = data['specific_angular_momentum_x_bins']
        specific_angular_momentum_y_bins   = data['specific_angular_momentum_y_bins']
        specific_angular_momentum_z_bins   = data['specific_angular_momentum_z_bins']
        temperature_bins                   = data['temperature_bins']

        if i == len(files)-1:
            data={}
            data['azimuthal_velocity_Mass']               =        azimuthal_velocity_Mass/len(files)
            data['azimuthal_velocity_entropy_Mass']       =        azimuthal_velocity_entropy_Mass/len(files)
            data['azimuthal_velocity_temperature_Mass']   =        azimuthal_velocity_temperature_Mass/len(files)
            data['density_temperature_Mass']              =        density_temperature_Mass/len(files)
            data['entropy_Mass']                          =        entropy_Mass/len(files)
            data['number_density_Mass']                   =        number_density_Mass/len(files)
            data['polar_velocity_Mass']                   =        polar_velocity_Mass/len(files)
            data['pressure_Mass']                         =        pressure_Mass/len(files)
            data['pressure_entropy_Mass']                 =        pressure_entropy_Mass/len(files)
            data['radial_velocity_Mass']                  =        radial_velocity_Mass/len(files)
            data['radial_velocity_entropy_Mass']          =        radial_velocity_entropy_Mass/len(files)
            data['radial_velocity_temperature_Mass']      =        radial_velocity_temperature_Mass/len(files)
            data['specific_angular_momentum_x_Mass']      =        specific_angular_momentum_x_Mass/len(files)
            data['specific_angular_momentum_y_Mass']      =        specific_angular_momentum_y_Mass/len(files)
            data['specific_angular_momentum_z_Mass']      =        specific_angular_momentum_z_Mass/len(files)
            # data['tcool_Mass']                            =        tcool_Mass/len(files)
            data['temperature_Mass']                      =        temperature_Mass/len(files)
            data['azimuthal_velocity_Volume']             =        azimuthal_velocity_Volume/len(files)
            data['azimuthal_velocity_entropy_Volume']     =        azimuthal_velocity_entropy_Volume/len(files)
            data['azimuthal_velocity_temperature_Volume'] =        azimuthal_velocity_temperature_Volume/len(files)
            data['density_temperature_Volume']            =        density_temperature_Volume/len(files)
            data['entropy_Volume']                        =        entropy_Volume/len(files)
            data['number_density_Volume']                 =        number_density_Volume/len(files)
            data['polar_velocity_Volume']                 =        polar_velocity_Volume/len(files)
            data['pressure_Volume']                       =        pressure_Volume/len(files)
            data['pressure_entropy_Volume']               =        pressure_entropy_Volume/len(files)
            data['radial_velocity_Volume']                =        radial_velocity_Volume/len(files)
            data['radial_velocity_entropy_Volume']        =        radial_velocity_entropy_Volume/len(files)
            data['radial_velocity_temperature_Volume']    =        radial_velocity_temperature_Volume/len(files)
            data['specific_angular_momentum_x_Volume']    =        specific_angular_momentum_x_Volume/len(files)
            data['specific_angular_momentum_y_Volume']    =        specific_angular_momentum_y_Volume/len(files)
            data['specific_angular_momentum_z_Volume']    =        specific_angular_momentum_z_Volume/len(files)
            # data['tcool_Volume']                          =        tcool_Volume/len(files)
            data['temperature_Volume']                    =        temperature_Volume/len(files)

            data['r_r200m_phase'] = r_r200m_phase
            data['r_r200m_profile'] = r_r200m_profile
            data['halo_mass'] = halo_mass
            data['time'] = time
            data['r200m'] = r200m
            data['azimuthal_velocity_bins'] = azimuthal_velocity_bins
            data['entropy_bins'] = entropy_bins
            data['number_density_bins'] = number_density_bins
            data['polar_velocity_bins'] = polar_velocity_bins
            data['pressure_bins'] = pressure_bins
            data['radial_velocity_bins'] = radial_velocity_bins
            data['specific_angular_momentum_x_bins'] = specific_angular_momentum_x_bins
            data['specific_angular_momentum_y_bins'] = specific_angular_momentum_y_bins
            data['specific_angular_momentum_z_bins'] = specific_angular_momentum_z_bins
            data['temperature_bins'] = temperature_bins
    return data

































































    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################

    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################

    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################


def new_plotter(data,fn, palet, sSFR_title=False):
    dlogT = np.diff(np.log10(data['temperature_bins']))[0]
    dlogn = np.diff(np.log10(data['number_density_bins']))[0]
    dlogP = np.diff(np.log10(data['pressure_bins']))[0]
    dlogK = np.diff(np.log10(data['entropy_bins']))[0]
    dlogr = np.diff(np.log10(data['r_r200m_profile']))[0]
    dvr = np.diff(data['radial_velocity_bins'])
    dvphi = np.diff(data['azimuthal_velocity_bins'])


    if len(data['r_r200m_profile']) > 200:
        r_r200m_profile_centers = 10**(np.log10(data['r_r200m_profile'])[:-1] + 0.5*np.diff(np.log10(data['r_r200m_profile'])))
    else :
        r_r200m_profile_centers = data['r_r200m_profile']
    temperature_bin_centers = 10**(np.log10(data['temperature_bins'])[:-1] + 0.5*np.diff(np.log10(data['temperature_bins'])))
    number_density_bin_centers = 10**(np.log10(data['number_density_bins'])[:-1] + 0.5*np.diff(np.log10(data['number_density_bins'])))
    entropy_bin_centers = 10**(np.log10(data['entropy_bins'])[:-1] + 0.5*np.diff(np.log10(data['entropy_bins'])))
    pressure_bin_centers = 10**(np.log10(data['pressure_bins'])[:-1] + 0.5*np.diff(np.log10(data['pressure_bins'])))
    radial_velocity_bin_centers = data['radial_velocity_bins'][:-1] + 0.5*np.diff(data['radial_velocity_bins'])
    azimuthal_velocity_bin_centers = data['azimuthal_velocity_bins'][:-1] + 0.5*np.diff(data['azimuthal_velocity_bins'])


    Nradii = len(r_r200m_profile_centers)
    sigma = 0.75

    temperature_profile_Volume = np.zeros(Nradii)*np.nan
    temperature_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['temperature_Volume'][:,i]) > 0:
            temperature_profile_Volume[i] = 10**np.average(np.log10(temperature_bin_centers), weights=data['temperature_Volume'][:,i])
        if np.sum(data['temperature_Mass'][:,i]) > 0:
            temperature_profile_Mass[i] = 10**np.average(np.log10(temperature_bin_centers), weights=data['temperature_Mass'][:,i])

    number_density_profile_Volume = np.zeros(Nradii)*np.nan
    number_density_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['number_density_Volume'][:,i]) > 0:
            number_density_profile_Volume[i] = 10**np.average(np.log10(number_density_bin_centers), weights=data['number_density_Volume'][:,i])
        if np.sum(data['number_density_Mass'][:,i]) > 0:
            number_density_profile_Mass[i] = 10**np.average(np.log10(number_density_bin_centers), weights=data['number_density_Mass'][:,i])

    entropy_profile_Volume = np.zeros(Nradii)*np.nan
    entropy_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['entropy_Volume'][:,i]) > 0:
            entropy_profile_Volume[i] = 10**np.average(np.log10(entropy_bin_centers), weights=data['entropy_Volume'][:,i])
        if np.sum(data['entropy_Mass'][:,i]) > 0:
            entropy_profile_Mass[i] = 10**np.average(np.log10(entropy_bin_centers), weights=data['entropy_Mass'][:,i])
    
    pressure_profile_Volume = np.zeros(Nradii)*np.nan
    pressure_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['pressure_Volume'][:,i]) > 0:
            pressure_profile_Volume[i] = 10**np.average(np.log10(pressure_bin_centers), weights=data['pressure_Volume'][:,i])
        if np.sum(data['pressure_Mass'][:,i]) > 0:
            pressure_profile_Mass[i] = 10**np.average(np.log10(pressure_bin_centers), weights=data['pressure_Mass'][:,i])

    radial_velocity_profile_Volume = np.zeros(Nradii)*np.nan
    radial_velocity_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['radial_velocity_Volume'][:,i]) > 0:
            radial_velocity_profile_Volume[i] = np.average(radial_velocity_bin_centers, weights=data['radial_velocity_Volume'][:,i])
        if np.sum(data['radial_velocity_Mass'][:,i]) > 0:
            radial_velocity_profile_Mass[i] = np.average(radial_velocity_bin_centers, weights=data['radial_velocity_Mass'][:,i])

    azimuthal_velocity_profile_Volume = np.zeros(Nradii)*np.nan
    azimuthal_velocity_profile_Mass = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['azimuthal_velocity_Volume'][:,i]) > 0:
            azimuthal_velocity_profile_Volume[i] = np.average(azimuthal_velocity_bin_centers, weights=data['azimuthal_velocity_Volume'][:,i])
        if np.sum(data['azimuthal_velocity_Mass'][:,i]) > 0:
            azimuthal_velocity_profile_Mass[i] = np.average(azimuthal_velocity_bin_centers, weights=data['azimuthal_velocity_Mass'][:,i])



    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
    #     (data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)),#(data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, temperature_bin_centers, 
        gaussian_filter((data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.T(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.T(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, temperature_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, temperature_profile_Volume, lw=2.5, label=r'$\langle T \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(T,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/temperature_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/temperature_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['temperature_bins'], 
    #     (data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)),#(data['temperature_Volume']/np.sum(data['temperature_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, temperature_bin_centers, 
        gaussian_filter((data['temperature_Mass']/np.sum(data['temperature_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.T(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.T(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.T(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, temperature_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, temperature_profile_Mass, lw=2.5, label=r'$\langle T \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(T,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_ylim(3e3,4e7)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/temperature_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/temperature_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
    #     (data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)),#(data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, number_density_bin_centers, 
        gaussian_filter((data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.n(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.n(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, number_density_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, number_density_profile_Volume, lw=2.5, label=r'$\langle n \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(n,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/number_density_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/number_density_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['number_density_bins'], 
    #     (data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)),#(data['number_density_Volume']/np.sum(data['number_density_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, number_density_bin_centers, 
        gaussian_filter((data['number_density_Mass']/np.sum(data['number_density_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.n(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.n(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.n(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, number_density_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, number_density_profile_Mass, lw=2.5, label=r'$\langle n \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(n,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim(5e-6,1e-1)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/number_density_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/number_density_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()



    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
    #     (data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)),#(data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, pressure_bin_centers, 
        gaussian_filter((data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.P(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.P(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, pressure_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, pressure_profile_Volume, lw=2.5, label=r'$\langle P \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(P,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/pressure_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/pressure_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['pressure_bins'], 
    #     (data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)),#(data['pressure_Volume']/np.sum(data['pressure_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, pressure_bin_centers, 
        gaussian_filter((data['pressure_Mass']/np.sum(data['pressure_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.P(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.P(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.P(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, pressure_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, pressure_profile_Mass, lw=2.5, label=r'$\langle P \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(P,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim(1e0,1e5)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/pressure_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/pressure_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()



    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
    #     (data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)),#(data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, entropy_bin_centers, 
        gaussian_filter((data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.K(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.K(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, entropy_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, entropy_profile_Volume, lw=2.5, label=r'$\langle K \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(K,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/entropy_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/entropy_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['entropy_bins'], 
    #     (data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)),#(data['entropy_Volume']/np.sum(data['entropy_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, entropy_bin_centers, 
        gaussian_filter((data['entropy_Mass']/np.sum(data['entropy_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_halo.K(radii)         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.K(radii)         , lw=2.25  , color = '1.0', alpha=0.5)
    ax.plot(radii/(rvir*kpc), HSE_adiabat_halo.K(radii)         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, entropy_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, entropy_profile_Mass, lw=2.5, label=r'$\langle K \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(K,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='lower right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/entropy_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/entropy_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
    #     (data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)),#(data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, radial_velocity_bin_centers, 
        gaussian_filter((data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, radial_velocity_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, radial_velocity_profile_Volume, lw=2.5, label=r'$\langle v_r \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(v_r,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/radial_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['radial_velocity_bins'], 
    #     (data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)),#(data['radial_velocity_Volume']/np.sum(data['radial_velocity_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, radial_velocity_bin_centers, 
        gaussian_filter((data['radial_velocity_Mass']/np.sum(data['radial_velocity_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, radial_velocity_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, radial_velocity_profile_Mass, lw=2.5, label=r'$\langle v_r \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(v_r,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/radial_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
    #     (data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)),#(data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, azimuthal_velocity_bin_centers, 
        gaussian_filter((data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, azimuthal_velocity_profile_Volume, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, azimuthal_velocity_profile_Volume, lw=2.5, label=r'$\langle v_\phi \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1])
    cb.set_label(r'${\rm Vol}(v_\phi,r)/{\rm Vol}(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/azimuthal_velocity_Volume_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/azimuthal_velocity_Volume_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


    fig,ax = plt.subplots(1,1)
    # plot=ax.pcolormesh(data['r_r200m_profile'], data['azimuthal_velocity_bins'], 
    #     (data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)),#(data['azimuthal_velocity_Volume']/np.sum(data['azimuthal_velocity_Volume'],axis=0)), 
    #     norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    plot=ax.contourf(r_r200m_profile_centers, azimuthal_velocity_bin_centers, 
        gaussian_filter((data['azimuthal_velocity_Mass']/np.sum(data['azimuthal_velocity_Mass'],axis=0)),sigma), 
        levels=np.append(np.logspace(-4,-0.5,61),10000),
        norm=colors.LogNorm(vmin=1e-4,vmax=10**-0.5), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_halo.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $T=T_{\rm vir}(r)$'   , color = '0.2', alpha=0.5)
    
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=2.25  , color = '1.0', alpha=0.5)
    # ax.plot(radii/(rvir*kpc), cooling_flow_haladiabat_o.vr(radii)/1e5         , lw=1.5, label=r'HSeq. $K=K_{\rm vir}$'   , color = '0.2', alpha=0.5, dashes=[4,2])

    ax.plot(r_r200m_profile_centers, azimuthal_velocity_profile_Mass, lw=3.25, color = 'white')
    ax.plot(r_r200m_profile_centers, azimuthal_velocity_profile_Mass, lw=2.5, label=r'$\langle v_\phi \rangle$'   , color = 'k')

    # ax.plot(radii/(rvir*kpc), HSE_halo_Kvir.T(radii)    , lw=2.5, label=r'HSE $K=K_{\rm vir}(r)$'   , color = '0.5', alpha=0.75, ls='--')
    cb = fig.colorbar(plot, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both")
    cb.set_label(r'$M(v_\phi,r)/M(r)$',rotation=270,fontsize=12,labelpad=15)
    ax.set_ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    # ax.set_ylim(1e5,3e10)
    ax.set_xlim(5e-2, 1)
    ax.set_xlabel(r'$r/r_{\rm vir}$')
    ax.legend(loc='upper right', fontsize=8,ncol=1,columnspacing=-3, handlelength=3.0)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/azimuthal_velocity_Mass_'+fn+'_linear.png',bbox_inches='tight',dpi=400)
    ax.set_xscale('log')
    plt.savefig('./plots/azimuthal_velocity_Mass_'+fn+'.png',bbox_inches='tight',dpi=400)
    plt.clf()


        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################


    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,2:4],axis=-1)/np.sum(data['pressure_entropy_Mass'][...,2:4]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(P,K,0.2\leq r/r_{\rm vir} \leq 0.4) / M(0.2\leq r/r_{\rm vir} \leq 0.4)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))

    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,2:4],axis=(1,2))/np.sum(data['pressure_entropy_Mass'][...,2:4]),sigma)/dlogK,data['entropy_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    fig.set_size_inches(5,4)
    plt.savefig('./plots/pressure_entropy_Mass_02r04_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ############################################################################################################################################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,6:8],axis=-1)/np.sum(data['pressure_entropy_Mass'][...,6:8]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(P,K,0.6\leq r/r_{\rm vir} \leq 0.8) / M(0.6\leq r/r_{\rm vir} \leq 0.8)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))

    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,6:8],axis=(1,2))/np.sum(data['pressure_entropy_Mass'][...,6:8]),sigma)/dlogK,data['entropy_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/pressure_entropy_Mass_06r08_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['pressure_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['pressure_entropy_Mass'][...,1:10]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(P,K,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim((7e-1,2e4))

    ir = 1
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir]),sigma)/dlogK,data['entropy_bins'][:-1], color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir]),sigma)/dlogK,data['entropy_bins'][:-1], color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir]),sigma)/dlogK,data['entropy_bins'][:-1], color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir]),sigma)/dlogK,data['entropy_bins'][:-1], color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['pressure_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['pressure_entropy_Mass'][...,ir]/np.sum(data['pressure_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,ir],axis=(1))/np.sum(data['pressure_entropy_Mass'][...,ir]),sigma)/dlogK,data['entropy_bins'][:-1], color='0.8', lw=2.)

    ax.legend(loc='lower left', fontsize=6, ncol=1)


    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/pressure_entropy_Mass_all_r_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['number_density_bins'], data['temperature_bins'], 
        gaussian_filter(np.sum(data['density_temperature_Mass'][...,2:4],axis=-1)/np.sum(data['density_temperature_Mass'][...,2:4]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(n,T,0.2\leq r/r_{\rm vir} \leq 0.4) / M(0.2\leq r/r_{\rm vir} \leq 0.4)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim((3e3,1e7))
    ax.set_xlim((3e-6,0.8e-1))

    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,2:4],axis=(1,2))/np.sum(data['density_temperature_Mass'][...,2:4]),sigma)/dlogT,data['temperature_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((3e3,1e7))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log T}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/density_temperature_Mass_02r04_'+fn+'_with_dMdlogT.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################
    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['number_density_bins'], data['temperature_bins'], 
        gaussian_filter(np.sum(data['density_temperature_Mass'][...,6:8],axis=-1)/np.sum(data['density_temperature_Mass'][...,6:8]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(n,T,0.6\leq r/r_{\rm vir} \leq 0.8) / M(0.6\leq r/r_{\rm vir} \leq 0.8)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim((3e3,1e7))
    ax.set_xlim((3e-6,0.8e-1))

    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,6:8],axis=(1,2))/np.sum(data['density_temperature_Mass'][...,6:8]),sigma)/dlogT,data['temperature_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((3e3,1e7))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log T}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/density_temperature_Mass_06r08_'+fn+'_with_dMdlogT.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['number_density_bins'], data['temperature_bins'], 
        gaussian_filter(np.sum(data['density_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['density_temperature_Mass'][...,1:10]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(n,T,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
    ax.set_ylim((3e3,1e7))
    ax.set_xlim((3e-6,0.8e-1))

    ir = 1
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir]),sigma)/dlogT,data['temperature_bins'][:-1], color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir]),sigma)/dlogT,data['temperature_bins'][:-1], color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir]),sigma)/dlogT,data['temperature_bins'][:-1], color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir]),sigma)/dlogT,data['temperature_bins'][:-1], color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['number_density_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['density_temperature_Mass'][...,ir]/np.sum(data['density_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axR.plot(gaussian_filter(np.sum(data['density_temperature_Mass'][...,ir],axis=(1))/np.sum(data['density_temperature_Mass'][...,ir]),sigma)/dlogT,data['temperature_bins'][:-1], color='0.8', lw=2.)

    ax.legend(loc='lower left', fontsize=6, ncol=1)


    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((3e3,1e7))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log T}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/density_temperature_Mass_all_r_'+fn+'_with_dMdlogT.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################
        ############################################################################################################################################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['azimuthal_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10]),sigma),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,K)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.0", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.2", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.4", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.6", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="0.8", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e4,1e10))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/azimuthal_velocity_entropy_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       

    ##############################################################################################################


    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['azimuthal_velocity_bins'], data['temperature_bins'], 
        gaussian_filter(np.sum(data['azimuthal_velocity_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_temperature_Mass'][...,1:10]),sigma),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.jim_special_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_\phi,T)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='#6600cc',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#6600cc", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='#0066ff',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#0066ff", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='#66cc00',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#66cc00", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='#ff9900',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#ff9900", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_temperature_Mass'][...,ir]/np.sum(data['azimuthal_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='#cc3300',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color="#cc3300", label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e3,3e7))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/azimuthal_velocity_temperature_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['radial_velocity_bins'], data['temperature_bins'], 
        gaussian_filter(np.sum(data['radial_velocity_temperature_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_temperature_Mass'][...,1:10]),sigma),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,T)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['temperature_bins'][:-1], 
        gaussian_filter((data['radial_velocity_temperature_Mass'][...,ir]/np.sum(data['radial_velocity_temperature_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower right', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e3,3e7))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$T\,[\mathrm{K}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_temperature_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       


    ##############################################################################################################

    fig,ax = plt.subplots(1,1)
    plot=ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,1:10]),sigma),
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=palet)#palettable.cubehelix.red_16_r.mpl_colormap)
    cb = fig.colorbar(plot)
    cb.set_label(r'$M(v_r,K)/M_{\rm tot}$',rotation=270,fontsize=12,labelpad=15)

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')

    ax.legend(loc='lower right', fontsize=6, ncol=1)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim((1e4,3e10))
    # ax.set_xlim((7e-1,2e4))
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,3)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_entropy_Mass_all_r_'+fn+'.png',bbox_inches='tight',dpi=200)
    plt.clf()       

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(80, 65, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[20:80, :60])
    axT= plt.subplot( gs[0:20, :60])
    cax = plt.subplot(gs[20:80, 63:])

    plot = ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,1:10]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='vertical', cax=cax)
    cb.set_label(r'$M(K,v_r,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, rotation=270, labelpad=15)
    # cb.ax.minorticks_off()
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,3e10))
    ax.set_xlim((-300,600))

    ir = 1
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir]),sigma)/np.diff(data['radial_velocity_bins']), color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir]),sigma)/np.diff(data['radial_velocity_bins']), color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir]),sigma)/np.diff(data['radial_velocity_bins']), color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir]),sigma)/np.diff(data['radial_velocity_bins']), color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['radial_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['radial_velocity_entropy_Mass'][...,ir]/np.sum(data['radial_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['radial_velocity_bins'][:-1],gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,ir],axis=(0))/np.sum(data['radial_velocity_entropy_Mass'][...,ir]),sigma)/np.diff(data['radial_velocity_bins']), color='0.8', lw=2.)

    ax.legend(loc='lower right', fontsize=6, ncol=1)


    axT.set_yscale('log')
    axT.set_xticklabels([])

    axT.set_ylim((1e-5,1e-1))
    axT.set_xlim((-300,600))
    axT.yaxis.set_label_position("right")
    axT.set_ylabel(r'$\frac{d \log M}{ d v_r}$', fontsize=14, rotation=270, labelpad=30)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(4,5)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_entropy_Mass_all_r_'+fn+'_with_dMdvr.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ############################################################################################################################################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,2:4],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,2:4]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(v_r,K,0.2\leq r/r_{\rm vir} \leq 0.4) / M(0.2\leq r/r_{\rm vir} \leq 0.4)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')

    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim(right=700)

    axR.plot(gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,2:4],axis=(1,2))/np.sum(data['radial_velocity_entropy_Mass'][...,2:4]),sigma)/dlogK,data['entropy_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_entropy_Mass_02r04_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ############################################################################################################################################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(65, 80, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[5:80, :60])
    axR= plt.subplot(gs[5:80, 63:])
    cax = plt.subplot(gs[0:3, :60])

    plot = ax.pcolormesh(data['radial_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,6:8],axis=-1)/np.sum(data['radial_velocity_entropy_Mass'][...,6:8]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='horizontal', cax=cax)
    cb.ax.set_title(r'$M(v_r,K,0.6\leq r/r_{\rm vir} \leq 0.8) / M(0.6\leq r/r_{\rm vir} \leq 0.8)$', fontsize=10, loc='left')
    # cb.ax.minorticks_off()
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')

    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_r\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,1e10))
    ax.set_xlim(right=700)

    axR.plot(gaussian_filter(np.sum(data['radial_velocity_entropy_Mass'][...,6:8],axis=(1,2))/np.sum(data['radial_velocity_entropy_Mass'][...,6:8]),sigma)/dlogK,data['entropy_bins'][:-1], color='k', lw=2.)
    axR.set_yscale('log')
    axR.set_xscale('log')
    axR.set_ylim((1e4,1e10))
    axR.set_yticklabels([])
    # axR.set_xticks([])

    axR.set_xlim((1e-4,10))

    axR.set_xlabel(r'$\frac{d \log M}{ d \log K}$', fontsize=14)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(5,4)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/radial_velocity_entropy_Mass_06r08_'+fn+'_with_dMdlogK.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    ##############################################################################################################

    fig= plt.figure(1)
    gs = gridspec.GridSpec(80, 65, wspace=0, hspace=0.0)
    ax = plt.subplot( gs[20:80, :60])
    axT= plt.subplot( gs[0:20, :60])
    cax = plt.subplot(gs[20:80, 63:])

    plot = ax.pcolormesh(data['azimuthal_velocity_bins'], data['entropy_bins'], 
        gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10],axis=-1)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,1:10]),sigma),
        cmap=palet,#cmap=palettable.cubehelix.red_16_r.mpl_colormap, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1e-1),rasterized=True)
    cb = fig.colorbar(plot, orientation='vertical', cax=cax)
    cb.set_label(r'$M(K,v_\phi,0.1\leq r/r_{\rm vir} \leq 1) / M(0.1\leq r/r_{\rm vir} \leq 1)$', fontsize=10, rotation=270, labelpad=15)
    # cb.ax.minorticks_off()
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=10) 
    ax.set_yscale('log')
    ax.set_ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    ax.set_xlabel(r'$v_\phi\,[\mathrm{km/s}]$')
    ax.set_ylim((1e4,3e10))
    ax.set_xlim((-300,300))

    ir = 1
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1],
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.0',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.0', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0)),sigma)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.0', lw=2.)

    ir = 2
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.2',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.2', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0)),sigma)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.2', lw=2.)

    ir = 3
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.4',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.4', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0)),sigma)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.4', lw=2.)

    ir = 5
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.6',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.6', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0)),sigma)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.6', lw=2.)

    ir = 7
    plot=ax.contour(data['azimuthal_velocity_bins'][:-1], data['entropy_bins'][:-1], 
        gaussian_filter((data['azimuthal_velocity_entropy_Mass'][...,ir]/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])),sigma),
        [3e-3], colors='0.8',norm=colors.LogNorm(),antialiased=True)
    ax.plot([1e-2,1e-2], [1e11,1e11], color='0.8', label=r'$'+str( np.round((data['r_r200m_phase']-np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( np.round( (data['r_r200m_phase']+np.diff(data['r_r200m_phase'])[0]/2.)[ir],1) )+r'$')
    axT.plot(data['azimuthal_velocity_bins'][:-1],gaussian_filter(np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir],axis=(0)),sigma)/np.sum(data['azimuthal_velocity_entropy_Mass'][...,ir])/np.diff(data['azimuthal_velocity_bins']), color='0.8', lw=2.)

    ax.legend(loc='lower right', fontsize=6, ncol=1)


    axT.set_yscale('log')
    axT.set_xticklabels([])

    axT.set_ylim((1e-5,1e-1))
    axT.set_xlim((-300,300))
    axT.yaxis.set_label_position("right")
    axT.set_ylabel(r'$\frac{d \log M}{ d v_\phi}$', fontsize=14, rotation=270, labelpad=30)
    ax.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    fig.set_size_inches(4,5)
    if sSFR_title:
        ax.text(0.05,0.95, r"$\log {\rm \, sSFR} = %s$" % np.round(np.log10(data['sSFR']),2), ha="left", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})
    plt.savefig('./plots/azimuthal_velocity_entropy_Mass_all_r_'+fn+'_with_dMdvphi.png', bbox_inches='tight',dpi=200)
    plt.close('all')
    plt.clf()  

    plt.close("all")













    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################

    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################

    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################
    #############################################################################################################################################################################################################











files = np.sort(glob.glob("./data/simulations/ksu*/*npz"))
for fn in files:
    data = np.load(fn)
    new_plotter(data,fn.split('/')[-1][:-4],palettable.cubehelix.jim_special_16_r.mpl_colormap)



TNG100_quenched_palet = palettable.cubehelix.Cubehelix.make(                             start=0.75, rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_quenched/*npz'))
data = averager(files)
new_plotter(data, "daniel_M12_TNG100_quenched_averaged",TNG100_quenched_palet)

Single_TNG100_Q_palet = palettable.cubehelix.Cubehelix.make(                             start=0.75+(1/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
fn = './data/simulations/daniel_M12_TNG100_quenched/Subhalo536830_snap099_TNG100.npz'
data = np.load(fn)
new_plotter(data,"Subhalo536830_snap099_TNG100_Quenched",Single_TNG100_Q_palet)

drummond_M12_var_palet = palettable.cubehelix.Cubehelix.make(                            start=0.75+(2/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/drummond/*drummond*var*npz'))
data = averager(files)
new_plotter(data, "drummond_M12_var_time_averaged",drummond_M12_var_palet)

MLi_SFR3_palet = palettable.cubehelix.Cubehelix.make(                                    start=0.75+(3/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR3*npz'))
data = averager(files)
new_plotter(data, "MLi_SFR3_time_averaged",MLi_SFR3_palet)



TNG100_starforming_palet = palettable.cubehelix.Cubehelix.make(                          start=2.75, rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_starforming/*npz'))
data = averager(files)
new_plotter(data, "daniel_M12_TNG100_starforming_averaged",TNG100_starforming_palet)

Single_TNG100_SF_palet = palettable.cubehelix.Cubehelix.make(                            start=2.75+(1/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
fn = './data/simulations/daniel_M12_TNG100_starforming/Subhalo518236_snap099_TNG100.npz'
data = np.load(fn)
new_plotter(data,"Subhalo518236_snap099_TNG100_StarForming",Single_TNG100_SF_palet)

drummond_M12_ref_palet = palettable.cubehelix.Cubehelix.make(                            start=2.75+(2/6.)-3.0, rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/drummond/*drummond*ref*npz'))
data = averager(files)
new_plotter(data, "drummond_M12_ref_time_averaged",drummond_M12_ref_palet)

MLi_SFR10_palet = palettable.cubehelix.Cubehelix.make(                                   start=2.75+(3/6.)-3.0, rotation=0., reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR10*npz'))
data = averager(files)
new_plotter(data, "MLi_SFR10_time_averaged",MLi_SFR10_palet)



files = np.sort(glob.glob("./data/simulations/daniel_M12_TNG100_starforming/*npz"))
for fn in files:
    data = np.load(fn)
    new_plotter(data,fn.split('/')[-1][:-4]+"_starforming",palettable.cubehelix.jim_special_16_r.mpl_colormap, sSFR_title=True)


files = np.sort(glob.glob("./data/simulations/daniel_M12_TNG100_quenched/*npz"))
for fn in files:
    data = np.load(fn)
    new_plotter(data,fn.split('/')[-1][:-4]+"_quenched",palettable.cubehelix.red_16_r.mpl_colormap, sSFR_title=True)