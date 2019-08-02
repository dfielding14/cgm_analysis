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



log_nHbins = np.arange(-8.,0.1,0.1)
log_Tbins  = np.array([ 2.        ,  2.01990533,  2.03977084,  2.05967712,  2.07957911,  2.09947348,  2.11935496,  2.13924932,  2.15911579,  2.17903447,  2.19890428,  2.21879792,  2.23869824,  2.25858951,  2.27847934,  2.29837275,  2.31825113,  2.33815765,  2.35804915,  2.37794328,  2.39781833,  2.41772079,  2.437608  ,  2.45750332,  2.47739625,  2.49728918,  2.51716948,  2.53706312,  2.55695343,  2.57684803,  2.59674001,  2.61663294,  2.636518  ,  2.65641451,  2.67630029,  2.69619918,  2.71608686,  2.73598194,  2.75586724,  2.77576327,  2.79565072,  2.81554461,  2.83543682,  2.85532522,  2.87521768,  2.89510751,  2.91499853,  2.93489218,  2.95478272,  2.97467279,  2.99456787,  3.01443648,  3.03434777,  3.05422997,  3.07412124,  3.09401655,  3.11390996,  3.13379431,  3.15369296,  3.17359424,  3.19348645,  3.2133584 ,  3.2332499 ,  3.25314403,  3.27304769,  3.29294252,  3.31283307,  3.33272123,  3.35260701,  3.37250686,  3.39239788,  3.41227579,  3.43216729,  3.45206261,  3.47195148,  3.49185181,  3.51173639,  3.53163218,  3.55152321,  3.57141757,  3.59130955,  3.61119199,  3.6310885 ,  3.65097737,  3.6708672 ,  3.690763  ,  3.71065044,  3.73054814,  3.75043893,  3.770329  ,  3.79022193,  3.81011152,  3.82999802,  3.84989214,  3.86978292,  3.88967705,  3.90956664,  3.92945981,  3.94935107,  3.96923876,  3.98913383,  4.00902557,  4.02889633,  4.04879141,  4.06870508,  4.08859682,  4.10846376,  4.12836695,  4.14826345,  4.16814375,  4.18805599,  4.20793056,  4.22783518,  4.24772787,  4.2676177 ,  4.2875104 ,  4.30738926,  4.32727718,  4.34717369,  4.3670578 ,  4.38696241,  4.40684652,  4.42673874,  4.44663048,  4.46652651,  4.48641634,  4.50630188,  4.52619696,  4.54608583,  4.56597757,  4.58586645,  4.60575724,  4.62565184,  4.64554977,  4.66544008,  4.68532944,  4.70522213,  4.72511101,  4.74500465,  4.76489305,  4.78478146,  4.80467749,  4.82456827,  4.84445858,  4.8643508 ,  4.88424015,  4.90413094,  4.92402601,  4.94391489,  4.96380663,  4.98369837,  5.00358963,  5.02349949,  5.04336214,  5.06325817,  5.08314419,  5.10305071,  5.12293625,  5.14282751,  5.16271353,  5.18261433,  5.2025156 ,  5.22240448,  5.24229288,  5.26216602,  5.28207827,  5.30196285,  5.32184696,  5.3417511 ,  5.3616333 ,  5.38153028,  5.40141773,  5.42130756,  5.44119215,  5.46109295,  5.48098373,  5.50086737,  5.52075863,  5.54065466,  5.5605402 ,  5.58043432,  5.60033035,  5.62021923,  5.64011335,  5.66000175,  5.67989111,  5.69978571,  5.71967936,  5.73956442,  5.75945616,  5.77935123,  5.79924393,  5.81913567,  5.8390255 ,  5.85891581,  5.87880898,  5.89869785,  5.91859102,  5.9384799 ,  5.95837259,  5.9782629 ,  5.99815464,  6.01803446,  6.03794432,  6.05781841,  6.07773113,  6.09760427,  6.11750317,  6.13738585,  6.1572752 ,  6.1771903 ,  6.19706011,  6.21695709,  6.23683929,  6.256742  ,  6.27662277,  6.29653358,  6.31641054,  6.33629942,  6.35619783,  6.37608385,  6.39597273,  6.41587448,  6.43576479,  6.45565176,  6.47554064,  6.49543333,  6.51533079,  6.53521824,  6.55510664,  6.57500315,  6.59488964,  6.61478138,  6.63467884,  6.65457106,  6.67445707,  6.69435072,  6.7142458 ,  6.73413563,  6.75402689,  6.77391815,  6.79380417,  6.81370115,  6.83359337,  6.85347939,  6.87337303,  6.89326239,  6.91315651,  6.93304682,  6.9529376 ,  6.97282743,  6.99272156,  7.01262617,  7.03249788,  7.05238628,  7.07228661,  7.09219408,  7.11206865,  7.13197136,  7.15185976,  7.17175579,  7.19164658,  7.21152115,  7.23141861,  7.25129747,  7.27119064,  7.29108   ,  7.31099033,  7.33088017,  7.35077095,  7.37066126,  7.39054632,  7.41043997,  7.43033314,  7.4502182 ,  7.47011614,  7.49000072,  7.50990105,  7.52978945,  7.54967737,  7.56957293,  7.58945799,  7.60934877,  7.62924623,  7.64913034,  7.66902828,  7.68891811,  7.70881176,  7.72870255,  7.74859095,  7.76848269,  7.78837347,  7.80826521,  7.82815695,  7.84804726,  7.86793852,  7.88783121,  7.90772295,  7.92761135,  7.94750214,  7.9673934 ,  7.98728609,  8.00719261,  8.02706432,  8.04696274,  8.0668478 ,  8.08675098,  8.10663319,  8.12652111,  8.14640713,  8.16631126,  8.18619347,  8.20609665,  8.22598076,  8.24588299,  8.26576138,  8.28564739,  8.30554485,  8.32543373,  8.34533405,  8.36522579,  8.38510513,  8.4050045 ,  8.42489815,  8.4447937 ,  8.46468353,  8.4845705 ,  8.50445747,  8.52435684,  8.54424191,  8.56413364,  8.58402538,  8.60391235,  8.62380695,  8.64369965,  8.66358757,  8.68347931,  8.70337772,  8.7232666 ,  8.74315643,  8.76304626,  8.782938  ,  8.80282879,  8.82272339,  8.84261513,  8.86250687,  8.88239384,  8.90228558,  8.92218018,  8.94206715,  8.96196175,  8.98185062])
redshifts  = np.array([0.000,0.049,0.101,0.155,0.211,0.271,0.333,0.399,0.468,0.540,0.615,0.695,0.778,0.957,1.154,1.370,1.609,1.871,2.013,2.160,2.479,2.829,3.017,3.214,3.638,4.356,4.895,5.184,5.807,6.141,6.859,7.650,8.521])
"""
Cooling curve as a function of density, temperature, metallicity, redshift
"""
Zs          = np.linspace(0,2,10)
file = glob.glob('./data/Cooling_Tables/Lambda_tab.npz')
if len(file) >0:
    a = np.load(file[0])
    Lambda_tab = a['Lambda_tab']
    Lambda      = interpolate.RegularGridInterpolator((log_nHbins,log_Tbins,Zs,redshifts), Lambda_tab, bounds_error=False, fill_value=0)
    a.close()
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
print 'interpolated lambda'




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
