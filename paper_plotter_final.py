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










def get_medians(data,fn):
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
    polar_velocity_bin_centers = azimuthal_velocity_bin_centers#data['polar_velocity_bins'][:-1] + 0.5*np.diff(data['polar_velocity_bins'])


    Nradii = len(r_r200m_profile_centers)
    sigma = 0.75

    temperature_profile_Volume = np.zeros(Nradii)*np.nan
    temperature_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    temperature_profile_Mass = np.zeros(Nradii)*np.nan
    temperature_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['temperature_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['temperature_Volume'][:,i])/np.sum(data['temperature_Volume'][:,i]),temperature_bin_centers, bounds_error=False, fill_value=0.0) 
            temperature_profile_Volume[i] =  f_interp(0.5)
            temperature_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            temperature_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['temperature_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['temperature_Mass'][:,i])/np.sum(data['temperature_Mass'][:,i]),temperature_bin_centers, bounds_error=False, fill_value=0.0) 
            temperature_profile_Mass[i] =  f_interp(0.5)
            temperature_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            temperature_profile_Mass_quartiles[1,i] =  f_interp( 0.75)

    number_density_profile_Volume = np.zeros(Nradii)*np.nan
    number_density_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    number_density_profile_Mass = np.zeros(Nradii)*np.nan
    number_density_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['number_density_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['number_density_Volume'][:,i])/np.sum(data['number_density_Volume'][:,i]),number_density_bin_centers, bounds_error=False, fill_value=0.0) 
            number_density_profile_Volume[i] =  f_interp(0.5)
            number_density_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            number_density_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['number_density_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['number_density_Mass'][:,i])/np.sum(data['number_density_Mass'][:,i]),number_density_bin_centers, bounds_error=False, fill_value=0.0) 
            number_density_profile_Mass[i] =  f_interp(0.5)
            number_density_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            number_density_profile_Mass_quartiles[1,i] =  f_interp( 0.75)

    entropy_profile_Volume = np.zeros(Nradii)*np.nan
    entropy_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    entropy_profile_Mass = np.zeros(Nradii)*np.nan
    entropy_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['entropy_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['entropy_Volume'][:,i])/np.sum(data['entropy_Volume'][:,i]),entropy_bin_centers, bounds_error=False, fill_value=0.0) 
            entropy_profile_Volume[i] =  f_interp(0.5)
            entropy_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            entropy_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['entropy_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['entropy_Mass'][:,i])/np.sum(data['entropy_Mass'][:,i]),entropy_bin_centers, bounds_error=False, fill_value=0.0) 
            entropy_profile_Mass[i] =  f_interp(0.5)
            entropy_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            entropy_profile_Mass_quartiles[1,i] =  f_interp( 0.75)
    
    pressure_profile_Volume = np.zeros(Nradii)*np.nan
    pressure_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    pressure_profile_Mass = np.zeros(Nradii)*np.nan
    pressure_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['pressure_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['pressure_Volume'][:,i])/np.sum(data['pressure_Volume'][:,i]),pressure_bin_centers, bounds_error=False, fill_value=0.0) 
            pressure_profile_Volume[i] =  f_interp(0.5)
            pressure_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            pressure_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['pressure_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['pressure_Mass'][:,i])/np.sum(data['pressure_Mass'][:,i]),pressure_bin_centers, bounds_error=False, fill_value=0.0) 
            pressure_profile_Mass[i] =  f_interp(0.5)
            pressure_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            pressure_profile_Mass_quartiles[1,i] =  f_interp( 0.75)

    radial_velocity_profile_Volume = np.zeros(Nradii)*np.nan
    radial_velocity_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    radial_velocity_profile_Mass = np.zeros(Nradii)*np.nan
    radial_velocity_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['radial_velocity_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['radial_velocity_Volume'][:,i])/np.sum(data['radial_velocity_Volume'][:,i]),radial_velocity_bin_centers, bounds_error=False, fill_value=0.0) 
            radial_velocity_profile_Volume[i] =  f_interp(0.5)
            radial_velocity_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            radial_velocity_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['radial_velocity_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['radial_velocity_Mass'][:,i])/np.sum(data['radial_velocity_Mass'][:,i]),radial_velocity_bin_centers, bounds_error=False, fill_value=0.0) 
            radial_velocity_profile_Mass[i] =  f_interp(0.5)
            radial_velocity_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            radial_velocity_profile_Mass_quartiles[1,i] =  f_interp( 0.75)

    azimuthal_velocity_profile_Volume = np.zeros(Nradii)*np.nan
    azimuthal_velocity_profile_Volume_quartiles = np.zeros((2,Nradii))*np.nan
    azimuthal_velocity_profile_Mass = np.zeros(Nradii)*np.nan
    azimuthal_velocity_profile_Mass_quartiles = np.zeros((2,Nradii))*np.nan
    for i in range(Nradii):
        if np.sum(data['azimuthal_velocity_Volume'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['azimuthal_velocity_Volume'][:,i])/np.sum(data['azimuthal_velocity_Volume'][:,i]),azimuthal_velocity_bin_centers, bounds_error=False, fill_value=0.0) 
            azimuthal_velocity_profile_Volume[i] =  f_interp(0.5)
            azimuthal_velocity_profile_Volume_quartiles[0,i] =  f_interp( 0.25)
            azimuthal_velocity_profile_Volume_quartiles[1,i] =  f_interp( 0.75)
        if np.sum(data['azimuthal_velocity_Mass'][:,i]) > 0:
            f_interp = scipy.interpolate.interp1d(np.cumsum(data['azimuthal_velocity_Mass'][:,i])/np.sum(data['azimuthal_velocity_Mass'][:,i]),azimuthal_velocity_bin_centers, bounds_error=False, fill_value=0.0) 
            azimuthal_velocity_profile_Mass[i] =  f_interp(0.5)
            azimuthal_velocity_profile_Mass_quartiles[0,i] =  f_interp( 0.25)
            azimuthal_velocity_profile_Mass_quartiles[1,i] =  f_interp( 0.75)




    medians={}
    medians["temperature_profile_Volume"] = temperature_profile_Volume
    medians["temperature_profile_Volume_quartiles"] = temperature_profile_Volume_quartiles
    medians["temperature_profile_Mass"] = temperature_profile_Mass
    medians["temperature_profile_Mass_quartiles"] = temperature_profile_Mass_quartiles
    medians["number_density_profile_Volume"] = number_density_profile_Volume
    medians["number_density_profile_Volume_quartiles"] = number_density_profile_Volume_quartiles
    medians["number_density_profile_Mass"] = number_density_profile_Mass
    medians["number_density_profile_Mass_quartiles"] = number_density_profile_Mass_quartiles
    medians["entropy_profile_Volume"] = entropy_profile_Volume
    medians["entropy_profile_Volume_quartiles"] = entropy_profile_Volume_quartiles
    medians["entropy_profile_Mass"] = entropy_profile_Mass
    medians["entropy_profile_Mass_quartiles"] = entropy_profile_Mass_quartiles
    medians["pressure_profile_Volume"] = pressure_profile_Volume
    medians["pressure_profile_Volume_quartiles"] = pressure_profile_Volume_quartiles
    medians["pressure_profile_Mass"] = pressure_profile_Mass
    medians["pressure_profile_Mass_quartiles"] = pressure_profile_Mass_quartiles
    medians["radial_velocity_profile_Volume"] = radial_velocity_profile_Volume
    medians["radial_velocity_profile_Volume_quartiles"] = radial_velocity_profile_Volume_quartiles
    medians["radial_velocity_profile_Mass"] = radial_velocity_profile_Mass
    medians["radial_velocity_profile_Mass_quartiles"] = radial_velocity_profile_Mass_quartiles
    medians["azimuthal_velocity_profile_Volume"] = azimuthal_velocity_profile_Volume
    medians["azimuthal_velocity_profile_Volume_quartiles"] = azimuthal_velocity_profile_Volume_quartiles
    medians["azimuthal_velocity_profile_Mass"] = azimuthal_velocity_profile_Mass
    medians["azimuthal_velocity_profile_Mass_quartiles"] = azimuthal_velocity_profile_Mass_quartiles


    medians["dMdlogK_02r04"] = gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,2:4],axis=(1,2))/np.sum(data['pressure_entropy_Mass'][...,2:4]),sigma)/dlogK
    medians["dMdlogK_06r08"] = gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,6:8],axis=(1,2))/np.sum(data['pressure_entropy_Mass'][...,6:8]),sigma)/dlogK

    medians["dMdlogP_02r04"] = gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,2:4],axis=(0,2))/np.sum(data['pressure_entropy_Mass'][...,2:4]),sigma)/dlogP
    medians["dMdlogP_06r08"] = gaussian_filter(np.sum(data['pressure_entropy_Mass'][...,6:8],axis=(0,2))/np.sum(data['pressure_entropy_Mass'][...,6:8]),sigma)/dlogP

    medians["dMdlogn_02r04"] = gaussian_filter(np.sum(data['density_temperature_Mass'][...,2:4],axis=(0,2))/np.sum(data['density_temperature_Mass'][...,2:4]),sigma)/dlogn
    medians["dMdlogn_06r08"] = gaussian_filter(np.sum(data['density_temperature_Mass'][...,6:8],axis=(0,2))/np.sum(data['density_temperature_Mass'][...,6:8]),sigma)/dlogn

    medians["dMdlogT_02r04"] = gaussian_filter(np.sum(data['density_temperature_Mass'][...,2:4],axis=(1,2))/np.sum(data['density_temperature_Mass'][...,2:4]),sigma)/dlogT
    medians["dMdlogT_06r08"] = gaussian_filter(np.sum(data['density_temperature_Mass'][...,6:8],axis=(1,2))/np.sum(data['density_temperature_Mass'][...,6:8]),sigma)/dlogT

    medians["dMdvr_02r04"] = gaussian_filter(np.sum(data['radial_velocity_temperature_Mass'][...,2:4],axis=(0,2))/np.sum(data['radial_velocity_temperature_Mass'][...,2:4]),sigma)
    medians["dMdvr_06r08"] = gaussian_filter(np.sum(data['radial_velocity_temperature_Mass'][...,6:8],axis=(0,2))/np.sum(data['radial_velocity_temperature_Mass'][...,6:8]),sigma)

    medians["Mdot_cold"] = np.array([ np.sum(np.sum(data['radial_velocity_temperature_Mass'][:20,:,iradius],axis=0) * radial_velocity_bin_centers*1e5 / (np.gradient(data['r_r200m_phase']*data['r200m'])[iradius] * kpc) * 3.15e7) for iradius in range(len(data['r_r200m_phase']))])
    medians["Mdot_warm"] = np.array([ np.sum(np.sum(data['radial_velocity_temperature_Mass'][20:30,:,iradius],axis=0) * radial_velocity_bin_centers*1e5 / (np.gradient(data['r_r200m_phase']*data['r200m'])[iradius] * kpc) * 3.15e7) for iradius in range(len(data['r_r200m_phase']))])
    medians["Mdot_hot"]  = np.array([ np.sum(np.sum(data['radial_velocity_temperature_Mass'][30:,:,iradius],axis=0) * radial_velocity_bin_centers*1e5 / (np.gradient(data['r_r200m_phase']*data['r200m'])[iradius] * kpc) * 3.15e7) for iradius in range(len(data['r_r200m_phase']))])
    medians["Mdot_radius"] = data['r_r200m_phase']
    medians["entropy_bin_centers"] = entropy_bin_centers
    medians["pressure_bin_centers"] = pressure_bin_centers
    medians["temperature_bin_centers"] = temperature_bin_centers
    medians["number_density_bin_centers"] = number_density_bin_centers
    medians["radial_velocity_bin_centers"] = radial_velocity_bin_centers


    radial_velocity_mean_MW = np.zeros(Nradii)*np.nan
    radial_velocity_std_MW  = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['radial_velocity_Mass'][:,i]) > 0:
            radial_velocity_mean_MW[i] = np.average(radial_velocity_bin_centers, weights = data['radial_velocity_Mass'][:,i])
            radial_velocity_std_MW[i] = np.sqrt(np.average( np.square(radial_velocity_bin_centers - radial_velocity_mean_MW[i]), weights = data['radial_velocity_Mass'][:,i]))

    azimuthal_velocity_mean_MW = np.zeros(Nradii)*np.nan
    azimuthal_velocity_std_MW  = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['azimuthal_velocity_Mass'][:,i]) > 0:
            azimuthal_velocity_mean_MW[i] = np.average(azimuthal_velocity_bin_centers, weights = data['azimuthal_velocity_Mass'][:,i])
            azimuthal_velocity_std_MW[i] = np.sqrt(np.average( np.square(azimuthal_velocity_bin_centers - azimuthal_velocity_mean_MW[i]), weights = data['azimuthal_velocity_Mass'][:,i]))

    polar_velocity_mean_MW = np.zeros(Nradii)*np.nan
    polar_velocity_std_MW  = np.zeros(Nradii)*np.nan
    for i in range(Nradii):
        if np.sum(data['polar_velocity_Mass'][:,i]) > 0:
            polar_velocity_mean_MW[i] = np.average(polar_velocity_bin_centers, weights = data['polar_velocity_Mass'][:,i])
            polar_velocity_std_MW[i] = np.sqrt(np.average( np.square(polar_velocity_bin_centers - polar_velocity_mean_MW[i]), weights = data['polar_velocity_Mass'][:,i]))

    medians["radial_velocity_std_MW"] = radial_velocity_std_MW
    medians["azimuthal_velocity_std_MW"] = azimuthal_velocity_std_MW
    medians["polar_velocity_std_MW"] = polar_velocity_std_MW
    medians["velocity_std_MW"] = np.sqrt( radial_velocity_std_MW**2 + azimuthal_velocity_std_MW**2 + polar_velocity_std_MW**2 )
    medians["r_r200m_profile_centers"] = r_r200m_profile_centers


    if type(data) == dict:
        if  "r200m" in data.keys():
            if data['halo_mass'] == 1e9:
                halo_mass = 1e12
            medians["r200m"] = data['r200m']
            medians["halo_mass"] = halo_mass
            nvir = halo_mass*2e33/(4*np.pi/3. * (data['r200m']*kpc)**3 * mu * mp)
            Tvir = 0.5*mu*mp*G*(halo_mass*2e33) / (data['r200m']*kpc) / kb
            medians["nvir"] = nvir
            medians["Pvir"] = nvir*Tvir
            medians["Tvir"] = Tvir
            medians["Kvir"] = nvir**(-2/3.) * Tvir
        else:
            medians["r200m"] = r200m
            medians["halo_mass"] = Mhalo/2e33
            nvir = halo_mass*2e33/(4*np.pi/3. * (r200m*kpc)**3 * mu * mp)
            Tvir = 0.5*mu*mp*G*(Mhalo) / (r200m*kpc) / kb
            medians["nvir"] = nvir
            medians["Pvir"] = nvir*Tvir
            medians["Tvir"] = Tvir
            medians["Kvir"] = nvir**(-2/3.) * Tvir
    else:
        if "r200m" in data.files:
            if data['halo_mass'] == 1e9:
                halo_mass = 1e12
            else:
                halo_mass = data['halo_mass']
            medians["r200m"] = data['r200m']
            medians["halo_mass"] = halo_mass
            nvir = halo_mass*2e33/(4*np.pi/3. * (data['r200m']*kpc)**3 * mu * mp)
            Tvir = 0.5*mu*mp*G*(halo_mass*2e33) / (data['r200m']*kpc) / kb
            medians["nvir"] = nvir
            medians["Pvir"] = nvir*Tvir
            medians["Tvir"] = Tvir
            medians["Kvir"] = nvir**(-2/3.) * Tvir
        else:
            if data['halo_mass'] == 1e9:
                halo_mass = 1e12
            medians["r200m"] = r200m
            medians["halo_mass"] = Mhalo/2e33
            nvir = halo_mass*2e33/(4*np.pi/3. * (r200m*kpc)**3 * mu * mp)
            Tvir = 0.5*mu*mp*G*(Mhalo) / (r200m*kpc) / kb
            medians["nvir"] = nvir
            medians["Pvir"] = nvir*Tvir
            medians["Tvir"] = Tvir
            medians["Kvir"] = nvir**(-2/3.) * Tvir

    print("done")

    return medians








files = np.sort(glob.glob("./data/simulations/ksu/average/*npz"))
ksu_medians={}
for fn in files:
    data = np.load(fn)
    ksu_medians[fn.split('/')[-1][:-4]] = get_medians(data,fn.split('/')[-1][:-4])




files = np.sort(glob.glob("./data/simulations/daniel_M12_TNG100_starforming/Sub*npz"))
M12_TNG100_starforming={}
for i,fn in enumerate(files):
    data = np.load(fn)
    M12_TNG100_starforming[fn.split('/')[-1].split('_')[0][7:]] = get_medians(data,fn.split('/')[-1][:-4]+"_starforming")
    print(fn.split('/')[-1].split('_')[0][7:])


files = np.sort(glob.glob("./data/simulations/daniel_M12_TNG100_quenched/Sub*npz"))
M12_TNG100_quenched={}
for i,fn in enumerate(files):
    data = np.load(fn)
    M12_TNG100_quenched[fn.split('/')[-1].split('_')[0][7:]]=get_medians(data,fn.split('/')[-1][:-4]+"_quenched")
    print(fn.split('/')[-1].split('_')[0][7:])


drummond_M12_var_palet = palettable.cubehelix.Cubehelix.make(                            start=0.75+(2/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/drummond/*drummond*var*npz'))
data = averager(files)
drummond_M12_var_time_averaged_medians = get_medians(data, "drummond_M12_var_time_averaged")

MLi_SFR3_palet = palettable.cubehelix.Cubehelix.make(                                    start=0.75+(3/6.), rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR3*npz'))
data = averager(files)
MLi_SFR3_time_averaged_medians = get_medians(data, "MLi_SFR3_time_averaged")

drummond_M12_ref_palet = palettable.cubehelix.Cubehelix.make(                            start=2.75+(2/6.)-3.0, rotation=0.15, reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/drummond/*drummond*ref*npz'))
data = averager(files)
drummond_M12_ref_time_averaged_medians = get_medians(data, "drummond_M12_ref_time_averaged")

MLi_SFR10_palet = palettable.cubehelix.Cubehelix.make(                                   start=2.75+(3/6.)-3.0, rotation=0., reverse=True, max_light=1.0, min_light=0.1, n=64).mpl_colormap
files = np.sort(glob.glob('./data/simulations/MLi/*MLi*_SFR10*npz'))
data = averager(files)
MLi_SFR10_time_averaged_medians = get_medians(data, "MLi_SFR10_time_averaged")







#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################





fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, sharex=True)
sigma_profiles=3.0
line_styles=['-','--',':']



ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['temperature_profile_Mass']/drummond_M12_var_time_averaged_medians['Tvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['temperature_profile_Mass']/drummond_M12_ref_time_averaged_medians['Tvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['temperature_profile_Mass']/MLi_SFR3_time_averaged_medians['Tvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['temperature_profile_Mass']/MLi_SFR10_time_averaged_medians['Tvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['temperature_profile_Mass']/ksu_medians[key]['Tvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


temperature_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['temperature_profile_Mass']/M12_TNG100_starforming[i]['Tvir'])):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['temperature_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['temperature_profile_Mass']),M12_TNG100_starforming[i]['temperature_profile_Mass']) / M12_TNG100_starforming[i]['Tvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            temperature_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_starforming += 1

temperature_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    temperature_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(temperature_profile_M12_TNG100_starforming_unshifted[:,k][temperature_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(temperature_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)


temperature_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['temperature_profile_Mass']/M12_TNG100_quenched[i]['Tvir'])):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['temperature_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['temperature_profile_Mass']),M12_TNG100_quenched[i]['temperature_profile_Mass']) / M12_TNG100_quenched[i]['Tvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            temperature_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

temperature_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    temperature_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(temperature_profile_M12_TNG100_quenched_unshifted[:,k][temperature_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(temperature_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax2.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['number_density_profile_Mass']/(drummond_M12_var_time_averaged_medians['nvir']*fb),
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['number_density_profile_Mass']/(drummond_M12_ref_time_averaged_medians['nvir']*fb),
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['number_density_profile_Mass']/(MLi_SFR3_time_averaged_medians['nvir']*fb),
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['number_density_profile_Mass']/(MLi_SFR10_time_averaged_medians['nvir']*fb),
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['number_density_profile_Mass']/(ksu_medians[key]['nvir']*fb),
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        



number_density_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['number_density_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['number_density_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['number_density_profile_Mass']),M12_TNG100_starforming[i]['number_density_profile_Mass']/(M12_TNG100_starforming[i]['nvir']*fb) )
    ax2.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                number_density_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        number_density_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

number_density_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    number_density_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(number_density_profile_M12_TNG100_starforming_unshifted[:,k][number_density_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax2.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(number_density_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




number_density_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['number_density_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['number_density_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['number_density_profile_Mass']),M12_TNG100_quenched[i]['number_density_profile_Mass']/(M12_TNG100_quenched[i]['nvir']*fb))
    ax2.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            number_density_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

number_density_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    number_density_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(number_density_profile_M12_TNG100_quenched_unshifted[:,k][number_density_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax2.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(number_density_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)





ax3.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['pressure_profile_Mass']/drummond_M12_var_time_averaged_medians['Pvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax3.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['pressure_profile_Mass']/drummond_M12_ref_time_averaged_medians['Pvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax3.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['pressure_profile_Mass']/MLi_SFR3_time_averaged_medians['Pvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax3.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['pressure_profile_Mass']/MLi_SFR10_time_averaged_medians['Pvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax3.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['pressure_profile_Mass']/ksu_medians[key]['Pvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

pressure_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['pressure_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['pressure_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['pressure_profile_Mass']),M12_TNG100_starforming[i]['pressure_profile_Mass']) /M12_TNG100_starforming[i]['Pvir']
    ax3.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                pressure_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        pressure_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

pressure_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    pressure_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(pressure_profile_M12_TNG100_starforming_unshifted[:,k][pressure_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax3.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(pressure_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




pressure_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['pressure_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['pressure_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['pressure_profile_Mass']),M12_TNG100_quenched[i]['pressure_profile_Mass'])/M12_TNG100_quenched[i]['Pvir']
    ax3.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            pressure_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

pressure_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    pressure_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(pressure_profile_M12_TNG100_quenched_unshifted[:,k][pressure_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax3.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(pressure_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)





ax4.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['entropy_profile_Mass']/drummond_M12_var_time_averaged_medians['Kvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax4.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['entropy_profile_Mass']/drummond_M12_ref_time_averaged_medians['Kvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax4.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['entropy_profile_Mass']/MLi_SFR3_time_averaged_medians['Kvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax4.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['entropy_profile_Mass']/MLi_SFR10_time_averaged_medians['Kvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax4.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['entropy_profile_Mass']/ksu_medians[key]['Kvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


entropy_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['entropy_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['entropy_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['entropy_profile_Mass']),M12_TNG100_starforming[i]['entropy_profile_Mass']) /M12_TNG100_starforming[i]['Kvir']
    ax4.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                entropy_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        entropy_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

entropy_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    entropy_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(entropy_profile_M12_TNG100_starforming_unshifted[:,k][entropy_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax4.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(entropy_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




entropy_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['entropy_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['entropy_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['entropy_profile_Mass']),M12_TNG100_quenched[i]['entropy_profile_Mass'])/M12_TNG100_quenched[i]['Kvir']
    ax4.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            entropy_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

entropy_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    entropy_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(entropy_profile_M12_TNG100_quenched_unshifted[:,k][entropy_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax4.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(entropy_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)





ax5.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['radial_velocity_profile_Mass']/(np.sqrt(G*drummond_M12_var_time_averaged_medians['halo_mass']*Msun/(drummond_M12_var_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax5.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['radial_velocity_profile_Mass']/(np.sqrt(G*drummond_M12_ref_time_averaged_medians['halo_mass']*Msun/(drummond_M12_ref_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax5.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['radial_velocity_profile_Mass']/(np.sqrt(G*MLi_SFR3_time_averaged_medians['halo_mass']*Msun/(MLi_SFR3_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax5.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['radial_velocity_profile_Mass']/(np.sqrt(G*MLi_SFR10_time_averaged_medians['halo_mass']*Msun/(MLi_SFR10_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax5.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['radial_velocity_profile_Mass']/(np.sqrt(G*ksu_medians[key]['halo_mass']*Msun/(ksu_medians[key]['r200m']*kpc))/1e5),
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        
radial_velocity_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']),M12_TNG100_starforming[i]['radial_velocity_profile_Mass']/(np.sqrt(G*M12_TNG100_starforming[i]['halo_mass']*Msun/(M12_TNG100_starforming[i]['r200m']*kpc))/1e5) ) 
    ax5.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                radial_velocity_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        radial_velocity_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

radial_velocity_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    radial_velocity_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(radial_velocity_profile_M12_TNG100_starforming_unshifted[:,k][radial_velocity_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax5.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(radial_velocity_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




radial_velocity_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']),M12_TNG100_quenched[i]['radial_velocity_profile_Mass']/(np.sqrt(G*M12_TNG100_quenched[i]['halo_mass']*Msun/(M12_TNG100_quenched[i]['r200m']*kpc))/1e5)
)
    ax5.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            radial_velocity_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

radial_velocity_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    radial_velocity_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(radial_velocity_profile_M12_TNG100_quenched_unshifted[:,k][radial_velocity_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax5.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(radial_velocity_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)






ax6.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['velocity_std_MW']/(np.sqrt(G*drummond_M12_var_time_averaged_medians['halo_mass']*Msun/(drummond_M12_var_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax6.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['velocity_std_MW']/(np.sqrt(G*drummond_M12_ref_time_averaged_medians['halo_mass']*Msun/(drummond_M12_ref_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax6.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['velocity_std_MW']/(np.sqrt(G*MLi_SFR3_time_averaged_medians['halo_mass']*Msun/(MLi_SFR3_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax6.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['velocity_std_MW']/(np.sqrt(G*MLi_SFR10_time_averaged_medians['halo_mass']*Msun/(MLi_SFR10_time_averaged_medians['r200m']*kpc))/1e5),
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax6.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['velocity_std_MW']/(np.sqrt(G*ksu_medians[key]['halo_mass']*Msun/(ksu_medians[key]['r200m']*kpc))/1e5),
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

velocity_std_MW_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['velocity_std_MW']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['velocity_std_MW']),M12_TNG100_starforming[i]['velocity_std_MW']/(np.sqrt(G*M12_TNG100_starforming[i]['halo_mass']*Msun/(M12_TNG100_starforming[i]['r200m']*kpc))/1e5) ) 
    ax6.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                velocity_std_MW_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        velocity_std_MW_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(velocity_std_MW_profile_M12_TNG100_starforming_unshifted[:,k][velocity_std_MW_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax6.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




velocity_std_MW_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['velocity_std_MW']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['velocity_std_MW']),M12_TNG100_quenched[i]['velocity_std_MW']/(np.sqrt(G*M12_TNG100_quenched[i]['halo_mass']*Msun/(M12_TNG100_quenched[i]['r200m']*kpc))/1e5))
    ax6.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            velocity_std_MW_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(velocity_std_MW_profile_M12_TNG100_quenched_unshifted[:,k][velocity_std_MW_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax6.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)





ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
# ax5.set_yscale('log')
# ax6.set_yscale('log')

ax1.set_xlim((0.095,1.05))


ax1.set_ylabel(r'$T/T_{\rm vir}$')
ax1.set_ylim(5e-3,3)

ax2.set_ylabel(r'$n/n_{\rm vir}$')
ax2.set_ylim((3e-2,1e3))

ax3.set_ylabel(r'$P / P_{\rm vir}$')
ax3.set_ylim((3e-3,1e1))

ax4.set_ylabel(r'$K / K_{\rm vir}$')
ax4.set_ylim((3e-4,3e1))
ax4.set_yticks([1e-3,1e-2,1e-1,1e0,1e1])

ax5.set_xlabel(r'$r/r_{200 m}$')
ax5.set_ylabel(r'$v_r/v_{\rm vir}$')
ax5.set_ylim((-1.02,0.52))

ax6.set_ylabel(r'$v_{\rm rms}/v_{\rm vir}$')
ax6.set_xlabel(r'$r/r_{200 m}$')
ax6.set_ylim((-0.02,1.52))



handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                 borderaxespad=0, ncol=4, fontsize=8, frameon=False)



ax2.yaxis.set_label_position("right")
ax4.yaxis.set_label_position("right")
ax6.yaxis.set_label_position("right")

ax2.yaxis.tick_right()
ax4.yaxis.tick_right()
ax6.yaxis.tick_right()

ax2.yaxis.set_ticks_position('both')
ax4.yaxis.set_ticks_position('both')
ax6.yaxis.set_ticks_position('both')


ax1.annotate('', xy=(0.2, 1e-2), xytext=(0.3, 1e-2), 
            arrowprops=dict(facecolor='black', shrink=0.,width=0.75,headwidth=2, headlength=3))

ax1.annotate('', xy=(0.4, 1e-2), xytext=(0.3, 1e-2), 
            arrowprops=dict(facecolor='black', shrink=0.,width=0.75,headwidth=2, headlength=3))

ax1.annotate('', xy=(0.6, 1e-2), xytext=(0.7, 1e-2), 
            arrowprops=dict(facecolor='black', shrink=0.,width=0.75,headwidth=2, headlength=3))

ax1.annotate('', xy=(0.8, 1e-2), xytext=(0.7, 1e-2), 
            arrowprops=dict(facecolor='black', shrink=0.,width=0.75,headwidth=2, headlength=3))

fig.set_size_inches(6.5,6.5)

plt.subplots_adjust(hspace=0.1,wspace=0.1)

plt.savefig('./plots/all_profs_MW.pdf',dpi=300,bbox_inches='tight')

ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4.set_xscale('log')
ax5.set_xscale('log')
ax6.set_xscale('log')
plt.savefig('./plots/all_profs_MW_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################









 



line_styles=['-','--',':']




sigma_histogram = 0.5


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_var_time_averaged_medians['dMdvr_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_ref_time_averaged_medians['dMdvr_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR3_time_averaged_medians['dMdvr_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR10_time_averaged_medians['dMdvr_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.semilogy(ksu_medians[key]['radial_velocity_bin_centers'],
                ksu_medians[key]['dMdvr_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdvr_02r04_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
        gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_02r04_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_02r04'][j]
    N_M12_TNG100_starforming += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdvr_02r04_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_02r04'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
        gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_02r04'][j]
    N_M12_TNG100_quenched += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')



ax1.set_ylabel(r'$dM / d v_r$')
# ax1.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((-220,220))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=4, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )



ax2.plot(drummond_M12_var_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_var_time_averaged_medians['dMdvr_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_ref_time_averaged_medians['dMdvr_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR3_time_averaged_medians['dMdvr_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR10_time_averaged_medians['dMdvr_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.semilogy(ksu_medians[key]['radial_velocity_bin_centers'],
                ksu_medians[key]['dMdvr_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)



dMdvr_06r08_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
        gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_06r08_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_06r08'][j]
    N_M12_TNG100_starforming += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdvr_06r08_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_06r08'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
        gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_06r08_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_06r08'][j]
    N_M12_TNG100_quenched += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)

        

ax2.set_ylabel(r'$dM / d v_r$')
ax2.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((-220,220))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdvr_unaligned.pdf',dpi=300,bbox_inches='tight')
plt.clf()











sigma_histogram = 0.5


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_var_time_averaged_medians['dMdvr_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_ref_time_averaged_medians['dMdvr_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR3_time_averaged_medians['dMdvr_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR10_time_averaged_medians['dMdvr_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.semilogy(ksu_medians[key]['radial_velocity_bin_centers'],
                ksu_medians[key]['dMdvr_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


dMdvr_02r04_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
        gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_02r04_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_02r04'][j]
    N_M12_TNG100_starforming += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdvr_02r04_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_02r04'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
        gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_02r04'][j]
    N_M12_TNG100_quenched += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d v_r$')
# ax1.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((-220,220))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)

l4 = ax1.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdvr_06r08_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0


ax2.plot(drummond_M12_var_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_var_time_averaged_medians['dMdvr_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['radial_velocity_bin_centers'],
            drummond_M12_ref_time_averaged_medians['dMdvr_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR3_time_averaged_medians['dMdvr_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['radial_velocity_bin_centers'],
            MLi_SFR10_time_averaged_medians['dMdvr_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.semilogy(ksu_medians[key]['radial_velocity_bin_centers'],
                ksu_medians[key]['dMdvr_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
        gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_06r08_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_06r08'][j]
    N_M12_TNG100_starforming += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdvr_06r08_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_06r08'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
        gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_06r08_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_06r08'][j]
    N_M12_TNG100_quenched += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)


ax2.set_ylabel(r'$dM / d v_r$')
ax2.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((-220,220))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(6.5,5)
fig.tight_layout()
plt.savefig('./plots/dMdvr_unaligned_big.pdf',dpi=300,bbox_inches='tight')
plt.clf()
































aligned_Pbins = 10**np.arange(-2.45, 5.46, 0.1) / 2000.


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['pressure_bin_centers']/drummond_M12_var_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogP_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogP_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['pressure_bin_centers']/drummond_M12_ref_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogP_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogP_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['pressure_bin_centers']/MLi_SFR3_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogP_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogP_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['pressure_bin_centers']/MLi_SFR10_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogP_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogP_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['pressure_bin_centers']/ksu_medians[key]['pressure_bin_centers'][np.argmax(ksu_medians[key]['dMdlogP_02r04'])],
                ksu_medians[key]['dMdlogP_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogP_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogP_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])], M12_TNG100_starforming[i]['dMdlogP_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogP_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogP_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])], M12_TNG100_quenched[i]['dMdlogP_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)

ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d\log P$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**2.5))
ax1.set_xticks([1e-2,1e-1,1e0,1e1,1e2])

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=4, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )


ax2.plot(drummond_M12_var_time_averaged_medians['pressure_bin_centers']/drummond_M12_var_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogP_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogP_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['pressure_bin_centers']/drummond_M12_ref_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogP_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogP_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['pressure_bin_centers']/MLi_SFR3_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogP_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogP_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['pressure_bin_centers']/MLi_SFR10_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogP_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogP_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['pressure_bin_centers']/ksu_medians[key]['pressure_bin_centers'][np.argmax(ksu_medians[key]['dMdlogP_06r08'])],
                ksu_medians[key]['dMdlogP_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        
dMdlogP_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogP_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])], M12_TNG100_starforming[i]['dMdlogP_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogP_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogP_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])], M12_TNG100_quenched[i]['dMdlogP_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)




ax2.set_ylabel(r'$dM / d\log P$')
ax2.set_xlabel(r'$P/P_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**2.5))
ax2.set_xticks([1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogP_aligned.pdf',dpi=300,bbox_inches='tight')
plt.clf()





































aligned_Pbins = 10**np.arange(-2.45, 5.46, 0.1) / 2000.


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['pressure_bin_centers']/drummond_M12_var_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogP_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogP_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['pressure_bin_centers']/drummond_M12_ref_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogP_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogP_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['pressure_bin_centers']/MLi_SFR3_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogP_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogP_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['pressure_bin_centers']/MLi_SFR10_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogP_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogP_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['pressure_bin_centers']/ksu_medians[key]['pressure_bin_centers'][np.argmax(ksu_medians[key]['dMdlogP_02r04'])],
                ksu_medians[key]['dMdlogP_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


dMdlogP_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogP_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])], M12_TNG100_starforming[i]['dMdlogP_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogP_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogP_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])], M12_TNG100_quenched[i]['dMdlogP_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)

ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d\log P$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**2.5))
ax1.set_xticks([1e-2,1e-1,1e0,1e1,1e2])

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )


ax2.plot(drummond_M12_var_time_averaged_medians['pressure_bin_centers']/drummond_M12_var_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogP_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogP_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['pressure_bin_centers']/drummond_M12_ref_time_averaged_medians['pressure_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogP_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogP_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['pressure_bin_centers']/MLi_SFR3_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogP_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogP_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['pressure_bin_centers']/MLi_SFR10_time_averaged_medians['pressure_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogP_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogP_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['pressure_bin_centers']/ksu_medians[key]['pressure_bin_centers'][np.argmax(ksu_medians[key]['dMdlogP_06r08'])],
                ksu_medians[key]['dMdlogP_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogP_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogP_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])], M12_TNG100_starforming[i]['dMdlogP_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogP_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogP_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])], M12_TNG100_quenched[i]['dMdlogP_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax2.set_ylabel(r'$dM / d\log P$')
ax2.set_xlabel(r'$P/P_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**2.5))
ax2.set_xticks([1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(6.5,5)
fig.tight_layout()
plt.savefig('./plots/dMdlogP_aligned_big.pdf',dpi=300,bbox_inches='tight')
plt.clf()





aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['entropy_bin_centers']/drummond_M12_var_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogK_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogK_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['entropy_bin_centers']/drummond_M12_ref_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogK_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogK_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['entropy_bin_centers']/MLi_SFR3_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogK_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogK_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['entropy_bin_centers']/MLi_SFR10_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogK_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogK_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['entropy_bin_centers']/ksu_medians[key]['entropy_bin_centers'][np.argmax(ksu_medians[key]['dMdlogK_02r04'])],
                ksu_medians[key]['dMdlogK_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogK_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogK_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])], M12_TNG100_starforming[i]['dMdlogK_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogK_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogK_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])], M12_TNG100_quenched[i]['dMdlogK_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)


ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.set_ylabel(r'$dM / d\log K$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((1e-4,1e2))
ax1.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=4, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )



ax2.plot(drummond_M12_var_time_averaged_medians['entropy_bin_centers']/drummond_M12_var_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogK_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogK_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['entropy_bin_centers']/drummond_M12_ref_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogK_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogK_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['entropy_bin_centers']/MLi_SFR3_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogK_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogK_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['entropy_bin_centers']/MLi_SFR10_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogK_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogK_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['entropy_bin_centers']/ksu_medians[key]['entropy_bin_centers'][np.argmax(ksu_medians[key]['dMdlogK_06r08'])],
                ksu_medians[key]['dMdlogK_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        



dMdlogK_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogK_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])], M12_TNG100_starforming[i]['dMdlogK_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogK_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogK_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])], M12_TNG100_quenched[i]['dMdlogK_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)


ax2.set_ylabel(r'$dM / d\log K$')
ax2.set_xlabel(r'$K/K_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((1e-4,1e2))
ax2.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogK_aligned.pdf',dpi=300,bbox_inches='tight')
plt.clf()












































aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['entropy_bin_centers']/drummond_M12_var_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogK_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogK_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['entropy_bin_centers']/drummond_M12_ref_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogK_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogK_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['entropy_bin_centers']/MLi_SFR3_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogK_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogK_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['entropy_bin_centers']/MLi_SFR10_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogK_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogK_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['entropy_bin_centers']/ksu_medians[key]['entropy_bin_centers'][np.argmax(ksu_medians[key]['dMdlogK_02r04'])],
                ksu_medians[key]['dMdlogK_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogK_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogK_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])], M12_TNG100_starforming[i]['dMdlogK_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogK_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogK_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])], M12_TNG100_quenched[i]['dMdlogK_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)


ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d\log K$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((1e-4,1e2))
ax1.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )



ax2.plot(drummond_M12_var_time_averaged_medians['entropy_bin_centers']/drummond_M12_var_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogK_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogK_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['entropy_bin_centers']/drummond_M12_ref_time_averaged_medians['entropy_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogK_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogK_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['entropy_bin_centers']/MLi_SFR3_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogK_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogK_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['entropy_bin_centers']/MLi_SFR10_time_averaged_medians['entropy_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogK_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogK_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['entropy_bin_centers']/ksu_medians[key]['entropy_bin_centers'][np.argmax(ksu_medians[key]['dMdlogK_06r08'])],
                ksu_medians[key]['dMdlogK_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogK_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogK_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])], M12_TNG100_starforming[i]['dMdlogK_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogK_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogK_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])], M12_TNG100_quenched[i]['dMdlogK_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)




ax2.set_ylabel(r'$dM / d\log K$')
ax2.set_xlabel(r'$K/K_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((1e-4,1e2))
ax2.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(6.5,5)
fig.tight_layout()
plt.savefig('./plots/dMdlogK_aligned_big.pdf',dpi=300,bbox_inches='tight')
plt.clf()




aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['temperature_bin_centers']/drummond_M12_var_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogT_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogT_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['temperature_bin_centers']/drummond_M12_ref_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogT_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogT_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['temperature_bin_centers']/MLi_SFR3_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogT_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogT_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['temperature_bin_centers']/MLi_SFR10_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogT_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogT_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['temperature_bin_centers']/ksu_medians[key]['temperature_bin_centers'][np.argmax(ksu_medians[key]['dMdlogT_02r04'])],
                ksu_medians[key]['dMdlogT_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogT_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_02r04'][20:])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogT_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogT_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_02r04'][20:])], M12_TNG100_starforming[i]['dMdlogT_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogT_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogT_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_02r04'][20:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogT_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogT_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_02r04'][20:])], M12_TNG100_quenched[i]['dMdlogT_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.set_ylabel(r'$dM / d\log T$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**1))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=4, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )





ax2.plot(drummond_M12_var_time_averaged_medians['temperature_bin_centers']/drummond_M12_var_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogT_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogT_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['temperature_bin_centers']/drummond_M12_ref_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogT_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogT_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['temperature_bin_centers']/MLi_SFR3_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogT_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogT_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['temperature_bin_centers']/MLi_SFR10_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogT_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogT_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['temperature_bin_centers']/ksu_medians[key]['temperature_bin_centers'][np.argmax(ksu_medians[key]['dMdlogT_06r08'])],
                ksu_medians[key]['dMdlogT_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


dMdlogT_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_06r08'][25:])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogT_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogT_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_06r08'][25:])], M12_TNG100_starforming[i]['dMdlogT_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogT_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogT_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_06r08'][25:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogT_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogT_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_06r08'][25:])], M12_TNG100_quenched[i]['dMdlogT_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax2.set_ylabel(r'$dM / d\log T$')
ax2.set_xlabel(r'$T/T_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**1))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogT_aligned.pdf',dpi=300,bbox_inches='tight')
plt.clf()

























aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['temperature_bin_centers']/drummond_M12_var_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogT_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogT_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['temperature_bin_centers']/drummond_M12_ref_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogT_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogT_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['temperature_bin_centers']/MLi_SFR3_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogT_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogT_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['temperature_bin_centers']/MLi_SFR10_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogT_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogT_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['temperature_bin_centers']/ksu_medians[key]['temperature_bin_centers'][np.argmax(ksu_medians[key]['dMdlogT_02r04'])],
                ksu_medians[key]['dMdlogT_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


dMdlogT_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_02r04'][20:])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogT_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogT_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_02r04'][20:])], M12_TNG100_starforming[i]['dMdlogT_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogT_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogT_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_02r04'][20:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogT_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogT_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][20:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_02r04'][20:])], M12_TNG100_quenched[i]['dMdlogT_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d\log T$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**1))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




ax2.plot(drummond_M12_var_time_averaged_medians['temperature_bin_centers']/drummond_M12_var_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogT_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogT_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['temperature_bin_centers']/drummond_M12_ref_time_averaged_medians['temperature_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogT_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogT_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['temperature_bin_centers']/MLi_SFR3_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogT_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogT_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['temperature_bin_centers']/MLi_SFR10_time_averaged_medians['temperature_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogT_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogT_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['temperature_bin_centers']/ksu_medians[key]['temperature_bin_centers'][np.argmax(ksu_medians[key]['dMdlogT_06r08'])],
                ksu_medians[key]['dMdlogT_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        



dMdlogT_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_06r08'][25:])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogT_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogT_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['temperature_bin_centers'] / M12_TNG100_starforming[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_starforming[i]['dMdlogT_06r08'][25:])], M12_TNG100_starforming[i]['dMdlogT_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogT_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogT_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_06r08'][25:])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogT_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogT_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['temperature_bin_centers'] / M12_TNG100_quenched[i]['temperature_bin_centers'][25:][np.argmax(M12_TNG100_quenched[i]['dMdlogT_06r08'][25:])], M12_TNG100_quenched[i]['dMdlogT_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogT_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax2.set_ylabel(r'$dM / d\log T$')
ax2.set_xlabel(r'$T/T_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**1))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(6.5,5)
fig.tight_layout()
plt.savefig('./plots/dMdlogT_aligned_big.pdf',dpi=300,bbox_inches='tight')
plt.clf()




aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['number_density_bin_centers']/drummond_M12_var_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogn_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogn_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['number_density_bin_centers']/drummond_M12_ref_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogn_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogn_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['number_density_bin_centers']/MLi_SFR3_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogn_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogn_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['number_density_bin_centers']/MLi_SFR10_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogn_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogn_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['number_density_bin_centers']/ksu_medians[key]['number_density_bin_centers'][np.argmax(ksu_medians[key]['dMdlogn_02r04'])],
                ksu_medians[key]['dMdlogn_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogn_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogn_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogn_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_02r04'])], M12_TNG100_starforming[i]['dMdlogn_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogn_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogn_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_02r04'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogn_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogn_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_02r04'])], M12_TNG100_quenched[i]['dMdlogn_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.set_ylabel(r'$dM / d\log n$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5, 10**2.5))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=4, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )



ax2.plot(drummond_M12_var_time_averaged_medians['number_density_bin_centers']/drummond_M12_var_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogn_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogn_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['number_density_bin_centers']/drummond_M12_ref_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogn_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogn_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['number_density_bin_centers']/MLi_SFR3_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogn_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogn_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['number_density_bin_centers']/MLi_SFR10_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogn_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogn_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['number_density_bin_centers']/ksu_medians[key]['number_density_bin_centers'][np.argmax(ksu_medians[key]['dMdlogn_06r08'])],
                ksu_medians[key]['dMdlogn_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        




dMdlogn_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogn_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogn_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_06r08'])], M12_TNG100_starforming[i]['dMdlogn_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogn_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogn_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_06r08'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogn_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogn_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_06r08'])], M12_TNG100_quenched[i]['dMdlogn_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)




ax2.set_ylabel(r'$dM / d\log n$')
ax2.set_xlabel(r'$n/n_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5, 10**2.5))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
plt.savefig('./plots/dMdlogn_aligned.pdf',dpi=300,bbox_inches='tight')
plt.clf()

























aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



ax1.plot(drummond_M12_var_time_averaged_medians['number_density_bin_centers']/drummond_M12_var_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogn_02r04'])],
            drummond_M12_var_time_averaged_medians['dMdlogn_02r04'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['number_density_bin_centers']/drummond_M12_ref_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogn_02r04'])],
            drummond_M12_ref_time_averaged_medians['dMdlogn_02r04'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')

ax1.plot(MLi_SFR3_time_averaged_medians['number_density_bin_centers']/MLi_SFR3_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogn_02r04'])],
            MLi_SFR3_time_averaged_medians['dMdlogn_02r04'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['number_density_bin_centers']/MLi_SFR10_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogn_02r04'])],
            MLi_SFR10_time_averaged_medians['dMdlogn_02r04'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)
ax1.plot(np.nan,np.nan,label = r' ',color='none')


for i,key in enumerate(ksu_medians.keys()):
    ax1.loglog(ksu_medians[key]['number_density_bin_centers']/ksu_medians[key]['number_density_bin_centers'][np.argmax(ksu_medians[key]['dMdlogn_02r04'])],
                ksu_medians[key]['dMdlogn_02r04'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

dMdlogn_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax1.plot(M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_02r04'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogn_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogn_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_02r04'])], M12_TNG100_starforming[i]['dMdlogn_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogn_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogn_02r04'])):
        continue
    ax1.plot(M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_02r04'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogn_02r04'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogn_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_02r04'])], M12_TNG100_quenched[i]['dMdlogn_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)

ax1.plot(np.nan,np.nan,label = r' ',color='none')


ax1.set_ylabel(r'$dM / d\log n$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5, 10**2.5))

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,3)
labels=np.roll(labels,3)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4, fontsize=10, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )





ax2.plot(drummond_M12_var_time_averaged_medians['number_density_bin_centers']/drummond_M12_var_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_var_time_averaged_medians['dMdlogn_06r08'])],
            drummond_M12_var_time_averaged_medians['dMdlogn_06r08'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax2.plot(drummond_M12_ref_time_averaged_medians['number_density_bin_centers']/drummond_M12_ref_time_averaged_medians['number_density_bin_centers'][np.argmax(drummond_M12_ref_time_averaged_medians['dMdlogn_06r08'])],
            drummond_M12_ref_time_averaged_medians['dMdlogn_06r08'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax2.plot(MLi_SFR3_time_averaged_medians['number_density_bin_centers']/MLi_SFR3_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR3_time_averaged_medians['dMdlogn_06r08'])],
            MLi_SFR3_time_averaged_medians['dMdlogn_06r08'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax2.plot(MLi_SFR10_time_averaged_medians['number_density_bin_centers']/MLi_SFR10_time_averaged_medians['number_density_bin_centers'][np.argmax(MLi_SFR10_time_averaged_medians['dMdlogn_06r08'])],
            MLi_SFR10_time_averaged_medians['dMdlogn_06r08'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax2.loglog(ksu_medians[key]['number_density_bin_centers']/ksu_medians[key]['number_density_bin_centers'][np.argmax(ksu_medians[key]['dMdlogn_06r08'])],
                ksu_medians[key]['dMdlogn_06r08'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        



dMdlogn_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    ax2.plot(M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_06r08'])] , 
        gaussian_filter(M12_TNG100_starforming[i]['dMdlogn_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)

    
    dMdlogn_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['number_density_bin_centers'] / M12_TNG100_starforming[i]['number_density_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogn_06r08'])], M12_TNG100_starforming[i]['dMdlogn_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




dMdlogn_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogn_06r08'])):
        continue
    ax2.plot(M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_06r08'])] , 
        gaussian_filter(M12_TNG100_quenched[i]['dMdlogn_06r08'], sigma_histogram),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    
    dMdlogn_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['number_density_bin_centers'] / M12_TNG100_quenched[i]['number_density_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogn_06r08'])], M12_TNG100_quenched[i]['dMdlogn_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogn_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax2.set_ylabel(r'$dM / d\log n$')
ax2.set_xlabel(r'$n/n_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5, 10**2.5))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(6.5,5)
plt.savefig('./plots/dMdlogn_aligned_big.pdf',dpi=300,bbox_inches='tight')
plt.clf()





























SFsatellites = np.load('data/simulations/daniel_M12_TNG100_starforming/SFsatellites.npz')
Qsatellites  = np.load('data/simulations/daniel_M12_TNG100_quenched/Qsatellites.npz')

SFsats = {}
for i in range(len(SFsatellites['subhaloID'])):
    SFsats[str(SFsatellites['subhaloID'][i])] = [SFsatellites['Nsats'][i],SFsatellites['Msats_total'][i],SFsatellites['Msats_gas'][i]]

Qsats = {}
for i in range(len(Qsatellites['subhaloID'])):
    Qsats[str(Qsatellites['subhaloID'][i])] = [Qsatellites['Nsats'][i],Qsatellites['Msats_total'][i],Qsatellites['Msats_gas'][i]]






fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.scatter(SFsatellites['Nsats'], SFsatellites['Msats_total'],color=palettable.cartocolors.qualitative.Bold_10.hex_colors[2], label='TNG SF')
ax1.scatter(Qsatellites['Nsats'], Qsatellites['Msats_total'],color=palettable.cartocolors.qualitative.Bold_10.hex_colors[4], label='TNG Q')
ax1.loglog()
ax1.legend(fontsize=8)

ax2.scatter(SFsatellites['Nsats'], SFsatellites['Msats_gas'],   color=palettable.cartocolors.qualitative.Bold_10.hex_colors[2])
ax2.scatter(Qsatellites['Nsats'], Qsatellites['Msats_gas'],     color=palettable.cartocolors.qualitative.Bold_10.hex_colors[4])
ax2.set_yscale('symlog', linthreshy=1e7)
ax2.set_xscale('log')
ax3.scatter(SFsatellites['Msats_total'], SFsatellites['Msats_gas'],color=palettable.cartocolors.qualitative.Bold_10.hex_colors[2])
ax3.scatter(Qsatellites['Msats_total'], Qsatellites['Msats_gas'],color=palettable.cartocolors.qualitative.Bold_10.hex_colors[4])
ax3.set_yscale('symlog', linthreshy=1e7)
ax3.set_xscale('log')

ax1.set_xlim((0,60))
ax1.set_ylim((3e9,1e12))

ax2.set_xlim((0,60))
ax2.set_ylim((-0.5e7,1e11))

ax3.set_xlim((3e9,1e12))
ax3.set_ylim((-0.5e7,1e11))


ax1.set_ylabel(r'$M_{sat, tot} \, [M_\odot]$')
ax2.set_ylabel(r'$M_{sat, gas} \, [M_\odot]$')
ax3.set_ylabel(r'$M_{sat, gas} \, [M_\odot]$')

ax1.set_xlabel(r'$N_{sat}$')
ax2.set_xlabel(r'$N_{sat}$')
ax3.set_xlabel(r'$M_{sat, tot} \, [M_\odot]$')

fig.set_size_inches((3.25,7))
fig.tight_layout()
plt.savefig('./plots/satellites_distribution.pdf',dpi=300,bbox_inches='tight')
plt.clf()







files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_quenched/Sub*npz'))
sSFR_quenched = []
for i,fn in enumerate(files):
    fn = fn[19:-4]
    data = np.load('./data/simulations/'+fn+'.npz')
    sSFR_quenched.append(data['sSFR'])

files = np.sort(glob.glob('./data/simulations/daniel_M12_TNG100_starforming/Sub*npz'))
sSFR_starforming = []
for i,fn in enumerate(files):
    fn = fn[19:-4]
    data = np.load('./data/simulations/'+fn+'.npz')
    sSFR_starforming.append(data['sSFR'])
    
sSFR_quenched = np.array(sSFR_quenched)
sSFR_quenched[sSFR_quenched==0] = 1e-14
sSFR_starforming = np.array(sSFR_starforming)

logsSFR_bins = np.arange(-15,-9.5,0.1)

fig,ax = plt.subplots(1,1)
ax.hist(np.log10(sSFR_starforming), logsSFR_bins, color=palettable.cartocolors.qualitative.Bold_10.hex_colors[2], label=r"TNG SF")
ax.hist(np.log10(sSFR_quenched), logsSFR_bins,    color=palettable.cartocolors.qualitative.Bold_10.hex_colors[4], label=r"TNG Q")
ax.set_yscale('symlog',linthreshy=10)
ax.set_ylabel(r'$N_{\rm gal}$')
ax.set_xlabel(r'$\log {\rm sSFR}$')
ax.set_yticks(np.append(np.arange(0,11,1),40))
labels = ["0","","2","","4","","6","","8","","10","40"]
ax.set_yticklabels(labels)
ax.set_xlim(left=-14.5)
labels = ["", 'no SF', "-13", "-12", "-11", "-10"]
ax.set_xticklabels(labels)
fig.set_size_inches(4,2.5)
plt.legend(loc='best')
plt.savefig('./plots/TNG100_sSFR_distribution.pdf', bbox_inches='tight',dpi=300)
plt.close('all')
plt.clf()  









sigma_histogram = 0.5


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdvr_02r04_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_02r04_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_02r04'][j]
    N_M12_TNG100_starforming += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_02r04_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_02r04'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue

    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] >= np.percentile(Qsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')


    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_02r04'][j]
    N_M12_TNG100_quenched += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d v_r$')
# ax1.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((-220,220))

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdvr_06r08_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_06r08_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_06r08'][j]
    N_M12_TNG100_starforming += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_06r08_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_06r08'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_06r08_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_06r08'][j]
    N_M12_TNG100_quenched += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d v_r$')
ax2.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((-220,220))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdvr_unaligned_Nsats.pdf',dpi=300,bbox_inches='tight')
plt.clf()




















aligned_Pbins = 10**np.arange(-2.45, 5.46, 0.1) / 2000.


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogP_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):

    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])], M12_TNG100_starforming[i]['dMdlogP_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_02r04'])):
        continue
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] >= np.percentile(Qsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])], M12_TNG100_quenched[i]['dMdlogP_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log P$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**2.5))
ax1.set_xticks([1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )


dMdlogP_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])], M12_TNG100_starforming[i]['dMdlogP_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_06r08'])):
        continue
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] >= np.percentile(Qsatellites["Nsats"],95):
        ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])], M12_TNG100_quenched[i]['dMdlogP_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log P$')
ax2.set_xlabel(r'$P/P_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**2.5))
ax2.set_xticks([1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogP_aligned_Nsats.pdf',dpi=300,bbox_inches='tight')
plt.clf()























aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogK_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])], M12_TNG100_starforming[i]['dMdlogK_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_02r04'])):
        continue
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] >= np.percentile(Qsatellites["Nsats"],95):
        ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])], M12_TNG100_quenched[i]['dMdlogK_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log K$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((1e-4,1e2))
ax1.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdlogK_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][0] <= np.percentile(SFsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][0] >= np.percentile(SFsatellites["Nsats"],95):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])], M12_TNG100_starforming[i]['dMdlogK_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_06r08'])):
        continue
    if Qsats[i][0] <= np.percentile(Qsatellites["Nsats"],5):
        ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][0] >= np.percentile(Qsatellites["Nsats"],95):
        ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])], M12_TNG100_quenched[i]['dMdlogK_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log K$')
ax2.set_xlabel(r'$K/K_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((1e-4,1e2))
ax2.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogK_aligned_Nsats.pdf',dpi=300,bbox_inches='tight')
plt.clf()
























sigma_histogram = 0.5


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdvr_02r04_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_02r04_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_02r04'][j]
    N_M12_TNG100_starforming += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_02r04_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_02r04'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue

    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')


    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_02r04'][j]
    N_M12_TNG100_quenched += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d v_r$')
# ax1.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((-220,220))

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdvr_06r08_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_06r08_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_06r08'][j]
    N_M12_TNG100_starforming += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_06r08_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_06r08'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_06r08_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_06r08'][j]
    N_M12_TNG100_quenched += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d v_r$')
ax2.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((-220,220))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdvr_unaligned_Msats_total.pdf',dpi=300,bbox_inches='tight')
plt.clf()




















aligned_Pbins = 10**np.arange(-2.45, 5.46, 0.1) / 2000.


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogP_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):

    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])], M12_TNG100_starforming[i]['dMdlogP_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_02r04'])):
        continue
    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])], M12_TNG100_quenched[i]['dMdlogP_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log P$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**2.5))
ax1.set_xticks([1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )


dMdlogP_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])], M12_TNG100_starforming[i]['dMdlogP_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_06r08'])):
        continue
    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])], M12_TNG100_quenched[i]['dMdlogP_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log P$')
ax2.set_xlabel(r'$P/P_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**2.5))
ax2.set_xticks([1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogP_aligned_Msats_total.pdf',dpi=300,bbox_inches='tight')
plt.clf()























aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogK_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])], M12_TNG100_starforming[i]['dMdlogK_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_02r04'])):
        continue
    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])], M12_TNG100_quenched[i]['dMdlogK_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log K$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((1e-4,1e2))
ax1.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdlogK_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][1] <= np.percentile(SFsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][1] >= np.percentile(SFsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])], M12_TNG100_starforming[i]['dMdlogK_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_06r08'])):
        continue
    if Qsats[i][1] <= np.percentile(Qsatellites["Msats_total"],5):
        ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][1] >= np.percentile(Qsatellites["Msats_total"],95):
        ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])], M12_TNG100_quenched[i]['dMdlogK_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log K$')
ax2.set_xlabel(r'$K/K_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((1e-4,1e2))
ax2.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogK_aligned_Msats_total.pdf',dpi=300,bbox_inches='tight')
plt.clf()




























sigma_histogram = 0.5


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdvr_02r04_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_02r04_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_02r04'][j]
    N_M12_TNG100_starforming += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_02r04_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_02r04'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue

    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            # gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')


    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_02r04'][j]
    N_M12_TNG100_quenched += 1


ax1.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_02r04_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d v_r$')
# ax1.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((-220,220))

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdvr_06r08_median_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_starforming[i]['radial_velocity_bin_centers'], 
            gaussian_filter(M12_TNG100_starforming[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_starforming[i]['radial_velocity_bin_centers'])):
        dMdvr_06r08_median_M12_TNG100_starforming_unshifted[II,j] +=  M12_TNG100_starforming[i]['dMdvr_06r08'][j]
    N_M12_TNG100_starforming += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_starforming_unshifted, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdvr_06r08_median_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_starforming.keys()),len(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdvr_06r08'])):
        dMdvr_02r04_median_M12_TNG100_quenched_unshifted[II]*=np.nan
        continue
    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            # gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] ,
            gaussian_filter(M12_TNG100_quenched[i]['dMdvr_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')

    for j in range(len(M12_TNG100_quenched[i]['radial_velocity_bin_centers'] )):
        dMdvr_06r08_median_M12_TNG100_quenched_unshifted[II,j] +=  M12_TNG100_quenched[i]['dMdvr_06r08'][j]
    N_M12_TNG100_quenched += 1


ax2.semilogy(M12_TNG100_starforming['478037']['radial_velocity_bin_centers'],
    np.nanmedian(dMdvr_06r08_median_M12_TNG100_quenched_unshifted, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d v_r$')
ax2.set_xlabel(r'$v_r \, [\mathrm{km/s}]$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((-220,220))
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdvr_unaligned_Msats_gas.pdf',dpi=300,bbox_inches='tight')
plt.clf()




















aligned_Pbins = 10**np.arange(-2.45, 5.46, 0.1) / 2000.


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogP_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):

    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_02r04'])], M12_TNG100_starforming[i]['dMdlogP_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_02r04'])):
        continue
    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            # gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_02r04'])], M12_TNG100_quenched[i]['dMdlogP_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log P$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((10**-2.5,10**2.5))
ax1.set_xticks([1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )


dMdlogP_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5, ls=':')

    
    dMdlogP_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['pressure_bin_centers'] / M12_TNG100_starforming[i]['pressure_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogP_06r08'])], M12_TNG100_starforming[i]['dMdlogP_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogP_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogP_06r08'])):
        continue
    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            # gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogP_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogP_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['pressure_bin_centers'] / M12_TNG100_quenched[i]['pressure_bin_centers'][np.argmax(M12_TNG100_quenched[i]['dMdlogP_06r08'])], M12_TNG100_quenched[i]['dMdlogP_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogP_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log P$')
ax2.set_xlabel(r'$P/P_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((10**-2.5,10**2.5))
ax2.set_xticks([1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogP_aligned_Msats_gas.pdf',dpi=300,bbox_inches='tight')
plt.clf()























aligned_Pbins = 10**np.arange(-4,4.01, 0.05)


fig,(ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=True)



dMdlogK_02r04_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_02r04_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_02r04'])], M12_TNG100_starforming[i]['dMdlogK_02r04'])
    N_M12_TNG100_starforming += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_02r04_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_02r04'])):
        continue
    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            # gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax1.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_02r04'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_02r04_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_02r04'][34:])], M12_TNG100_quenched[i]['dMdlogK_02r04'])
    N_M12_TNG100_quenched += 1


ax1.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_02r04_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax1.set_ylabel(r'$dM / d\log K$')
ax1.set_ylim(bottom=1e-3)
ax1.set_xlim((1e-4,1e2))
ax1.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])

ax1.plot([1,2],[1e-6,1e-6], lw=0.5, color='grey', label='few satellites')
ax1.plot([1,2],[1e-6,1e-6], lw=0.5, ls=':', color='grey', label='many satellites')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7, frameon=False)

ax1.text(0.95, 0.95, r"$0.2 \leq {r}/{r_{200m}} \leq 0.4$",
         ha="right", va="top", transform=ax1.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )




dMdlogK_06r08_median_M12_TNG100_starforming = np.zeros((len(M12_TNG100_starforming.keys()),len(aligned_Pbins)))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    if SFsats[i][2] <= np.percentile(SFsatellites["Msats_gas"],5):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5)
    if SFsats[i][2] >= np.percentile(SFsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])] , 
            gaussian_filter(M12_TNG100_starforming[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.5,ls=':')

    
    dMdlogK_06r08_median_M12_TNG100_starforming[II] += np.interp( aligned_Pbins , M12_TNG100_starforming[i]['entropy_bin_centers'] / M12_TNG100_starforming[i]['entropy_bin_centers'][np.argmax(M12_TNG100_starforming[i]['dMdlogK_06r08'])], M12_TNG100_starforming[i]['dMdlogK_06r08'])
    N_M12_TNG100_starforming += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_starforming, axis=0),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=0.5)




dMdlogK_06r08_median_M12_TNG100_quenched = np.zeros((len(M12_TNG100_quenched.keys()),len(aligned_Pbins)))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    if np.isnan(    np.max(M12_TNG100_quenched[i]['dMdlogK_06r08'])):
        continue
    # if Qsats[i][2] <= np.percentile(Qsatellites["Msats_gas"],5):
        # ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            # gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            # color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5)
    if Qsats[i][2] >= np.percentile(Qsatellites["Msats_gas"],95):
        ax2.plot(M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][34:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][34:])] , 
            gaussian_filter(M12_TNG100_quenched[i]['dMdlogK_06r08'], sigma_histogram),
            color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.5, ls=':')
    
    dMdlogK_06r08_median_M12_TNG100_quenched[II] += np.interp( aligned_Pbins , M12_TNG100_quenched[i]['entropy_bin_centers'] / M12_TNG100_quenched[i]['entropy_bin_centers'][42:][np.argmax(M12_TNG100_quenched[i]['dMdlogK_06r08'][42:])], M12_TNG100_quenched[i]['dMdlogK_06r08'])
    N_M12_TNG100_quenched += 1


ax2.loglog(aligned_Pbins,
    np.nanmedian(dMdlogK_06r08_median_M12_TNG100_quenched, axis=0),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=0.5)



ax2.set_ylabel(r'$dM / d\log K$')
ax2.set_xlabel(r'$K/K_{\rm max}$')
ax2.set_ylim(bottom=1e-3)
ax2.set_xlim((1e-4,1e2))
ax2.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.text(0.95, 0.95, r"$0.6 \leq {r}/{r_{200m}} \leq 0.8$",
         ha="right", va="top", transform=ax2.transAxes, fontsize=7,
         bbox=dict(boxstyle="round",ec='none',fc="0.25",alpha=0.25)
         )

fig.set_size_inches(3.25,4)
fig.tight_layout()
plt.savefig('./plots/dMdlogK_aligned_Msats_gas.pdf',dpi=300,bbox_inches='tight')
plt.clf()































































































































sigma_profiles=3.0



fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['temperature_profile_Mass']/drummond_M12_var_time_averaged_medians['Tvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['temperature_profile_Mass']/drummond_M12_ref_time_averaged_medians['Tvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['temperature_profile_Mass']/MLi_SFR3_time_averaged_medians['Tvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['temperature_profile_Mass']/MLi_SFR10_time_averaged_medians['Tvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['temperature_profile_Mass']/ksu_medians[key]['Tvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


temperature_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['temperature_profile_Mass']/M12_TNG100_starforming[i]['Tvir'])):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['temperature_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['temperature_profile_Mass']),M12_TNG100_starforming[i]['temperature_profile_Mass']) / M12_TNG100_starforming[i]['Tvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            temperature_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_starforming += 1

temperature_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    temperature_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(temperature_profile_M12_TNG100_starforming_unshifted[:,k][temperature_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(temperature_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




temperature_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['temperature_profile_Mass']/M12_TNG100_quenched[i]['Tvir'])):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['temperature_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['temperature_profile_Mass']),M12_TNG100_quenched[i]['temperature_profile_Mass']) / M12_TNG100_quenched[i]['Tvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            temperature_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

temperature_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    temperature_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(temperature_profile_M12_TNG100_quenched_unshifted[:,k][temperature_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(temperature_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$T/T_{\rm vir}$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)

fig.set_size_inches(3.25,3)
ax1.set_xlim((0.05,1))
ax1.set_ylim(5e-3,8)
ax1.semilogy()
plt.savefig('./plots/temperature_profile_Mass_all.pdf',dpi=300,bbox_inches='tight')
ax1.loglog()
plt.savefig('./plots/temperature_profile_Mass_all_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()












fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)

ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['number_density_profile_Mass'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['number_density_profile_Mass'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['number_density_profile_Mass'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['number_density_profile_Mass'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['number_density_profile_Mass'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        



number_density_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['number_density_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['number_density_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['number_density_profile_Mass']),M12_TNG100_starforming[i]['number_density_profile_Mass']) 
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                number_density_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        number_density_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

number_density_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    number_density_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(number_density_profile_M12_TNG100_starforming_unshifted[:,k][number_density_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(number_density_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




number_density_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['number_density_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['number_density_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['number_density_profile_Mass']),M12_TNG100_quenched[i]['number_density_profile_Mass'])
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            number_density_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

number_density_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    number_density_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(number_density_profile_M12_TNG100_quenched_unshifted[:,k][number_density_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(number_density_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$n\,[\mathrm{cm}^{-3}]$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)
fig.set_size_inches(3.25,3)
ax1.set_xlim((0.05,1))
ax1.set_ylim((1e-5,1e-1))
ax1.semilogy()
plt.savefig('./plots/number_density_profile_Mass_all.pdf',dpi=300,bbox_inches='tight')
ax1.loglog()
plt.savefig('./plots/number_density_profile_Mass_all_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()














fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['pressure_profile_Mass']/drummond_M12_var_time_averaged_medians['Pvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['pressure_profile_Mass']/drummond_M12_ref_time_averaged_medians['Pvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['pressure_profile_Mass']/MLi_SFR3_time_averaged_medians['Pvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['pressure_profile_Mass']/MLi_SFR10_time_averaged_medians['Pvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['pressure_profile_Mass']/ksu_medians[key]['Pvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

pressure_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['pressure_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['pressure_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['pressure_profile_Mass']),M12_TNG100_starforming[i]['pressure_profile_Mass']) /M12_TNG100_starforming[i]['Pvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                pressure_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        pressure_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

pressure_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    pressure_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(pressure_profile_M12_TNG100_starforming_unshifted[:,k][pressure_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(pressure_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




pressure_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['pressure_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['pressure_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['pressure_profile_Mass']),M12_TNG100_quenched[i]['pressure_profile_Mass'])/M12_TNG100_quenched[i]['Pvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            pressure_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

pressure_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    pressure_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(pressure_profile_M12_TNG100_quenched_unshifted[:,k][pressure_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(pressure_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)




ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$P / P_{\rm vir}$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)
fig.set_size_inches(3.25,3)
ax1.set_xlim((0.05,1))
# ax1.set_ylim((3e-4,3))
ax1.semilogy()
plt.savefig('./plots/pressure_profile_Mass_all_Pvir.pdf',dpi=300,bbox_inches='tight')
ax1.loglog()
plt.savefig('./plots/pressure_profile_Mass_all_Pvir_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()














fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)


ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['entropy_profile_Mass']/drummond_M12_var_time_averaged_medians['Kvir'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['entropy_profile_Mass']/drummond_M12_ref_time_averaged_medians['Kvir'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['entropy_profile_Mass']/MLi_SFR3_time_averaged_medians['Kvir'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['entropy_profile_Mass']/MLi_SFR10_time_averaged_medians['Kvir'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['entropy_profile_Mass']/ksu_medians[key]['Kvir'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        


entropy_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['entropy_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['entropy_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['entropy_profile_Mass']),M12_TNG100_starforming[i]['entropy_profile_Mass']) /M12_TNG100_starforming[i]['Kvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                entropy_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        entropy_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

entropy_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    entropy_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(entropy_profile_M12_TNG100_starforming_unshifted[:,k][entropy_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(entropy_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




entropy_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['entropy_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['entropy_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['entropy_profile_Mass']),M12_TNG100_quenched[i]['entropy_profile_Mass'])/M12_TNG100_quenched[i]['Kvir']
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            entropy_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

entropy_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    entropy_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(entropy_profile_M12_TNG100_quenched_unshifted[:,k][entropy_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(entropy_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$K / K_{\rm vir}$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)
fig.set_size_inches(3.25,3)
ax1.set_xlim((0.05,1))
ax1.semilogy()
plt.savefig('./plots/entropy_profile_Mass_all_Kvir.pdf',dpi=300,bbox_inches='tight')
ax1.loglog()
plt.savefig('./plots/entropy_profile_Mass_all_Kvir_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()










fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)




ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['radial_velocity_profile_Mass'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['radial_velocity_profile_Mass'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['radial_velocity_profile_Mass'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['radial_velocity_profile_Mass'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['radial_velocity_profile_Mass'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        
radial_velocity_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['radial_velocity_profile_Mass']),M12_TNG100_starforming[i]['radial_velocity_profile_Mass']) 
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                radial_velocity_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        radial_velocity_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

radial_velocity_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    radial_velocity_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(radial_velocity_profile_M12_TNG100_starforming_unshifted[:,k][radial_velocity_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(radial_velocity_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




radial_velocity_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['radial_velocity_profile_Mass']),M12_TNG100_quenched[i]['radial_velocity_profile_Mass'])
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            radial_velocity_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

radial_velocity_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    radial_velocity_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(radial_velocity_profile_M12_TNG100_quenched_unshifted[:,k][radial_velocity_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(radial_velocity_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)



ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$v_r\,[\mathrm{km/s}]$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)
fig.set_size_inches(3.25,3)
ax1.set_ylim((-150,150))
ax1.set_xlim((0.05,1))
plt.savefig('./plots/radial_velocity_profile_Mass_all.pdf',dpi=300,bbox_inches='tight')
ax1.semilogx()
plt.savefig('./plots/radial_velocity_profile_Mass_all_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()




























fig,(ax1) = plt.subplots(1,1, sharex=True, sharey=True)





ax1.plot(drummond_M12_var_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_var_time_averaged_medians['velocity_std_MW'],
            label = r"DF low eta",
            color = '0.0', lw=2, alpha=1.0)
ax1.plot(drummond_M12_ref_time_averaged_medians['r_r200m_profile_centers'],
            drummond_M12_ref_time_averaged_medians['velocity_std_MW'],
            label = r"DF high eta",
            color = '0.0',ls='--', lw=2, alpha=1.0)
ax1.plot(MLi_SFR3_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR3_time_averaged_medians['velocity_std_MW'],
            label = r"MLi SFR3",
            color = '0.66', lw=2, alpha=1.0)
ax1.plot(MLi_SFR10_time_averaged_medians['r_r200m_profile_centers'],
            MLi_SFR10_time_averaged_medians['velocity_std_MW'],
            label = r"MLi SFR10",
            color = '0.66',ls='--', lw=2, alpha=1.0)

for i,key in enumerate(ksu_medians.keys()):
    ax1.plot(ksu_medians[key]['r_r200m_profile_centers'],
                ksu_medians[key]['velocity_std_MW'],
                label = r"ksu "+key,
                color = '0.33',ls=line_styles[i], lw=2, alpha=1.0)
        

velocity_std_MW_profile_M12_TNG100_starforming_unshifted = np.zeros((len(M12_TNG100_starforming.keys()), len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])))
N_M12_TNG100_starforming = 0

for II,i in enumerate(M12_TNG100_starforming.keys()):
    # if np.isnan(    np.max(M12_TNG100_starforming[i]['velocity_std_MW']/M12_TNG100_starforming)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['velocity_std_MW']), M12_TNG100_starforming[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_starforming[i]['velocity_std_MW']),M12_TNG100_starforming[i]['velocity_std_MW']) 
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=0.25, alpha=0.25)
    if hasattr(T_Tvir.mask, '__len__'):
        for j in range(len(r_T)):
            if not T_Tvir.mask[j]:
                velocity_std_MW_profile_M12_TNG100_starforming_unshifted[II,j] +=  T_Tvir[j]
    else:
        velocity_std_MW_profile_M12_TNG100_starforming_unshifted[II] +=  T_Tvir
    N_M12_TNG100_starforming += 1

velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median = np.zeros(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_starforming['478037']['r_r200m_profile_centers'])):
    velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median[k] = np.median(velocity_std_MW_profile_M12_TNG100_starforming_unshifted[:,k][velocity_std_MW_profile_M12_TNG100_starforming_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_starforming[i]['r_r200m_profile_centers'],
    gaussian_filter(velocity_std_MW_profile_M12_TNG100_starforming_unshifted_median, sigma_profiles),
    label = r'TNG SF average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[2], lw=2, alpha=1.0)




velocity_std_MW_profile_M12_TNG100_quenched_unshifted = np.zeros((len(M12_TNG100_quenched.keys()), len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])))
N_M12_TNG100_quenched = 0

for II,i in enumerate(M12_TNG100_quenched.keys()):
    # if np.isnan(    np.max(M12_TNG100_quenched[i]['velocity_std_MW']/M12_TNG100_quenched)):
    #     continue
    r_T = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['velocity_std_MW']), M12_TNG100_quenched[i]['r_r200m_profile_centers'])
    T_Tvir = np.ma.masked_where(np.isnan(M12_TNG100_quenched[i]['velocity_std_MW']),M12_TNG100_quenched[i]['velocity_std_MW'])
    ax1.plot(r_T ,gaussian_filter(T_Tvir, sigma_profiles),
        color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=0.25, alpha=0.25)
    for j in range(len(r_T)):
        if not T_Tvir.mask[j]:
            velocity_std_MW_profile_M12_TNG100_quenched_unshifted[II,j] +=  T_Tvir[j]
    N_M12_TNG100_quenched += 1

velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median = np.zeros(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers']))
for k in range(len(M12_TNG100_quenched["482814"]['r_r200m_profile_centers'])):
    velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median[k] = np.median(velocity_std_MW_profile_M12_TNG100_quenched_unshifted[:,k][velocity_std_MW_profile_M12_TNG100_quenched_unshifted[:,k]!=0.])

ax1.plot(M12_TNG100_quenched[i]['r_r200m_profile_centers'],
    gaussian_filter(velocity_std_MW_profile_M12_TNG100_quenched_unshifted_median, sigma_profiles),
    label = r'TNG Q average',
    color = palettable.cartocolors.qualitative.Bold_10.hex_colors[4], lw=2, alpha=1.0)

ax1.set_xlabel(r'$r/r_{200 m}$')
ax1.set_ylabel(r'$\langle v^2 \rangle^{1/2}\,[\mathrm{km/s}]$')

handles, labels = ax1.get_legend_handles_labels()
handles=np.roll(handles,2)
labels=np.roll(labels,2)
l4 = ax1.legend(handles,labels,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3, fontsize=5.5)
fig.set_size_inches(3.25,3)
ax1.set_xlim((0.05,1))
plt.savefig('./plots/velocity_std_MW_all.pdf',dpi=300,bbox_inches='tight')
ax1.semilogx()
plt.savefig('./plots/velocity_std_MW_all_log.pdf',dpi=300,bbox_inches='tight')
plt.clf()
