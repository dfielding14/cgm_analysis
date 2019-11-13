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
import scipy.interpolate


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







def get_median_temperature(files):
    median = {}
    fn = files[0]
    fn = fn[19:-4]
    print(fn)
    data = np.load('./data/simulations/'+fn+'.npz')
    median['r_r200m_profile']   = data['r_r200m_profile']
    median['temperature_bins']  = data['temperature_bins']

    all_temperature_Mass = np.zeros((len(files), data['temperature_Mass'].shape[0], data['temperature_Mass'].shape[1] ))
    all_halo_mass = np.zeros((len(files)))
    all_r200m = np.zeros((len(files)))
    all_Tvir = np.zeros((len(files)))
    for i,fn in enumerate(files):
        fn = fn[19:-4]
        print(fn)
        data = np.load('./data/simulations/'+fn+'.npz')

        halo_mass = data['halo_mass']
        if halo_mass == 1e9:
            halo_mass = 1e12
        all_halo_mass[i]         = halo_mass
        all_r200m[i]             = data['r200m']
        Tvir = 0.5*mu*mp*G*(halo_mass*2e33) / (data['r200m']*kpc) / kb
        all_Tvir[i] = Tvir
        all_temperature_Mass[i] = gaussian_filter(data['temperature_Mass'] / Tvir,1.0)

    median['temperature_Mass'] = np.median(all_temperature_Mass,axis=0)*np.nanmedian(all_Tvir)
    median['halo_mass'] = np.nanmedian(all_halo_mass)
    median['r200m'] = np.nanmedian(all_r200m)
    median['Tvir'] = np.nanmedian(all_Tvir)

    return median

files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_quenched/Su*npz'))
TNG100_quenched_data = get_median_temperature(files)

files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_starforming/Su*npz'))
TNG100_starforming_data = get_median_temperature(files)

files = np.sort(glob.glob('./data/simulations/drummond/*drummond*var*npz'))
drummond_M12_var_data = averager(files)

files = np.sort(glob.glob('./data/simulations/drummond/*drummond*ref*npz'))
drummond_M12_ref_data = averager(files)

files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR3*npz'))
MLi_SFR3_data = averager(files)

files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR10*npz'))
MLi_SFR10_data = averager(files)

files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR3*npz'))
MLi_SFR3_data = averager(files)

files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR10*npz'))
MLi_SFR10_data = averager(files)

files = np.sort(glob.glob('./data/simulations/ksu/average/FIRE_only.npz'))
ksu_FIRE_data = np.load(files[0])

files = np.sort(glob.glob('./data/simulations/ksu/average/Thermal.npz'))
ksu_Thermal_data = np.load(files[0])

files = np.sort(glob.glob('./data/simulations/ksu/average/Turbulent.npz'))
ksu_Turbulent_data = np.load(files[0])





def plotter(data, ax, title):
    halo_mass = data['halo_mass'] 
    if halo_mass == 1e9:
        halo_mass = 1e12

    Tvir = 0.5*mu*mp*G*(halo_mass*2e33) / (data['r200m']*kpc) / kb
    print(np.log10(halo_mass), Tvir,data['r200m']) 

    if len(data['r_r200m_profile']) > 200:
        r_r200m_profile_centers = 10**(np.log10(data['r_r200m_profile'])[:-1] + 0.5*np.diff(np.log10(data['r_r200m_profile'])))
    else :
        r_r200m_profile_centers = data['r_r200m_profile']
    temperature_bin_centers = 10**(np.log10(data['temperature_bins'])[:-1] + 0.5*np.diff(np.log10(data['temperature_bins'])))

    Nradii = len(r_r200m_profile_centers)
    sigma = 0.75

    temperature_profile_Mass = np.zeros(Nradii)*np.nan
    temperature_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['temperature_Mass'][:,i]) > 0:
            temperature_profile_Mass[i] =  np.interp( 0.5, np.cumsum(data['temperature_Mass'][:,i])/np.sum(data['temperature_Mass'][:,i]),temperature_bin_centers) 
            temperature_profile_Mass_quartiles[0,i] =  np.interp( 0.25, np.cumsum(data['temperature_Mass'][:,i])/np.sum(data['temperature_Mass'][:,i]),temperature_bin_centers) 
            temperature_profile_Mass_quartiles[1,i] =  np.interp( 0.75, np.cumsum(data['temperature_Mass'][:,i])/np.sum(data['temperature_Mass'][:,i]),temperature_bin_centers) 


    T_smooth = gaussian_filter((data['temperature_Mass']/np.nansum(data['temperature_Mass'],axis=0)),sigma)
    # T_smooth[np.isnan(T_smooth)] = 0.0
    # T_smooth[T_smooth<1.01e-4] = 1.01e-4

    plot=ax.contourf(r_r200m_profile_centers, temperature_bin_centers/Tvir, T_smooth, 
        levels=np.append(np.logspace(-4,-0.5,100),10000),
        norm=colors.SymLogNorm(vmin=1e-4,vmax=10**-0.5,linthresh=1e-5), cmap="magma_r",zorder=-20)#palettable.cubehelix.jim_special_16_r.mpl_colormap)

    ax.loglog(r_r200m_profile_centers, gaussian_filter(temperature_profile_Mass/Tvir,2.0), lw=3.25, color = 'white')
    ax.loglog(r_r200m_profile_centers, gaussian_filter(temperature_profile_Mass/Tvir,2.0), lw=2.5, label=r'$\langle T \rangle$'   , color = 'k')

    ax.text(0.95,0.95, title, ha="right", va="top",transform=ax.transAxes, fontsize=8, bbox={'facecolor':'grey', 'edgecolor':'None', 'boxstyle':'round','pad':0.1, 'alpha':0.75})

    ax.set_ylim(5e-3,8)
    ax.set_xlim(0.95e-1, 1.05)

    return plot

fig, axarr = plt.subplots(3,3,sharex=True,sharey=True)

plot = plotter(TNG100_quenched_data,       axarr[1,0], 'TNG Q')
plot = plotter(TNG100_starforming_data,    axarr[0,0], 'TNG SF')
plot = plotter(drummond_M12_var_data,      axarr[1,1], 'DF low eta')
plot = plotter(drummond_M12_ref_data,      axarr[0,1], 'DF high eta')
plot = plotter(MLi_SFR3_data,              axarr[1,2], 'MLi SFR3')
plot = plotter(MLi_SFR10_data,             axarr[0,2], 'MLi SFR10')
plot = plotter(ksu_FIRE_data,              axarr[2,0], 'ksu FIRE')
plot = plotter(ksu_Thermal_data,           axarr[2,1], 'ksu Thermal')
plot = plotter(ksu_Turbulent_data,         axarr[2,2], 'ksu Turbulent')
# fig.tight_layout()
fig.subplots_adjust(hspace=0.1, wspace=0.1)
cax = fig.add_axes([0.15,0.,0.7,0.025])

cb = fig.colorbar(plot, cax=cax, ticks=[1e-4,1e-3,1e-2,1e-1],extend="both", orientation='horizontal')
cb.set_label(r'$M(T,r)/M(r)$')
axarr[1,0].set_ylabel(r"$T/T_{\rm vir}$")
axarr[0,0].set_ylabel(r"$T/T_{\rm vir}$")
axarr[2,0].set_ylabel(r"$T/T_{\rm vir}$")
axarr[2,0].set_xlabel(r"$r/r_{\rm 200m}$",)
axarr[2,1].set_xlabel(r"$r/r_{\rm 200m}$",)
axarr[2,2].set_xlabel(r"$r/r_{\rm 200m}$",)
axarr[2,0].set_xticklabels(['','','0.1','1'])
axarr[2,1].set_xticklabels(['','','0.1','1'])
axarr[2,2].set_xticklabels(['','','0.1','1'])

fig.set_size_inches(6.5,5.5)
plt.savefig('./plots/temperature_radius_all_sims_Mass_magma.png',bbox_inches='tight',dpi=400)
for j in range(3):
    for i in range(3):
        axarr[i,j].set_rasterization_zorder(-10)
plt.savefig('./plots/temperature_radius_all_sims_Mass_magma.pdf',bbox_inches='tight',dpi=400)
plt.clf()












