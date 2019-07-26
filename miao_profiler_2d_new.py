import matplotlib
matplotlib.rc('font', family='sans-serif', size=10)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['figure.figsize'] = 5,4
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from matplotlib import cm as CM
import yt
from yt.units import *
from yt.utilities.physical_constants import *
import matplotlib.colors as colors



import os

#### If you would like analyze many data outputs 
#### it will be much faster to do it in parallel

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

rank=1
size=1

##############################################################
####  Begin  Fielding+17 specific stuff to set the units  ####
##############################################################

drummond = False

if drummond:

    # read in and set up unit system
    H0 = 70 * km /s / Mpc
    mu = 0.62
    muH = 1/0.7
    Tcool =1e4
    athinput = open('athinput', 'r')
    for line in athinput:
        if "Tigm" in line: Tigm = float(line.strip()[12:])
        if "M15" in line: M15 = float(line.strip()[12:])
        if "c_nfw" in line: c = float(line.strip()[11:])
        if "z         =" in line: zz = float(line.strip()[11:])
        if "Tcool"       in line: Tcool = float(line.strip()[11:])
        if "num_domains" in line: nlevels = int(line.strip()[14:])
        if "METALLICITY" in line: METALLICITY = float(line.strip()[14:])
    athinput.close()
    OL = 0.73
    Om = 0.27

    Tcool *= K
    UnitLength = YTQuantity((G * M15 * 1e15 * Msun / H0**2)**(1./3.)).convert_to_units('kpc')
    UnitTime = YTQuantity(1/H0).convert_to_units('yr')
    UnitMass = YTQuantity(M15 * 1e15 * Msun).convert_to_units('Msun')
    UnitTemp = 4.688*0.62*M15**(2./3.) *keV
    kbTfloor = Tigm*UnitTemp
    H = np.sqrt(OL + (1+zz)**3 * Om)
    r200m = UnitLength * (H**2/( (1+zz)**3 * Om))**(1./3.)*(10*H)**(-2./3.)

    # read in and set up cooling curves
    H_He_Cooling  = np.loadtxt('H_He_cooling.dat')
    Tbins         = np.loadtxt('Tbins.dat')
    nHbins        = np.loadtxt('nHbins.dat')
    Metal_Cooling = np.loadtxt('Metal_cooling.dat')
    Metal_Cooling = METALLICITY*Metal_Cooling

    f_Metal_Cooling = interpolate.RegularGridInterpolator((np.log10(Tbins), np.log10(nHbins)),Metal_Cooling)
    f_H_He_Cooling  = interpolate.RegularGridInterpolator((np.log10(Tbins), np.log10(nHbins)), H_He_Cooling)
    f_Cooling       = interpolate.RegularGridInterpolator((np.log10(Tbins), np.log10(nHbins)),Metal_Cooling+H_He_Cooling, bounds_error=False, fill_value=1e-35)
    vf_Cooling = np.vectorize(f_Cooling)

    # make new yt fields for cooling
    def _cooling_function(field,data):
        logT  = np.log10(data['temperature'].d)
        lognH = np.log10(data['number_density_H'].d)
        # LAMBDA = vf_Cooling(logT, lognH) * erg*cm**3/s
        LAMBDA = f_Cooling((logT, lognH)) * erg*cm**3/s
        return LAMBDA
    yt.add_field(("gas", "cooling_function"), function=_cooling_function, units='erg*cm**3/s', display_name=r"$\Lambda$",force_override=True)
    yt.add_field(("gas", "lambda_field"), function=_cooling_function, units='erg*cm**3/s', display_name=r"$\Lambda$",force_override=True)

    def _tcool(field,data):
        logT  = np.log10(data['temperature'].d)
        lognH = np.log10(data['number_density_H'].d)
        # LAMBDA = vf_Cooling(logT, lognH) * erg*cm**3/s
        LAMBDA = f_Cooling((logT, lognH)) * erg*cm**3/s
        t_cool = 1.5*kb*data['temperature']*(muH/mu)**2/((data['density']/(mp*mu)) * LAMBDA)
        t_cool[np.isinf(t_cool)] = 0.0
        # t_cool[t_cool < 1e-6] = 1e6
        return t_cool
    yt.add_field(("gas","tcool"),function=_tcool,units="Gyr", display_name=r"$t_{\rm cool}$",force_override=True)

    def _edot_cool(field,data):
        logT  = np.log10(data['temperature'].d)
        lognH = np.log10(data['number_density_H'].d)
        # LAMBDA = vf_Cooling(logT, lognH) * erg*cm**3/s
        LAMBDA = f_Cooling((logT, lognH)) * erg*cm**3/s
        edot_cool = data['number_density_H']**2 * LAMBDA * data['cell_volume']
        return edot_cool
    yt.add_field(("gas","edot_cool"),function=_edot_cool,units="erg/s", display_name=r"$\dot{E}_{\rm cool}$",force_override=True)

    def tff(field,data):
        g = G*UnitMass/np.square(data['radius']) * (np.log(1.+c*data['radius']/r200m) - c*data['radius']/r200m/(1.+c*data['radius']/r200m))/(np.log(1.+c) - c/(1.+c))
        return np.sqrt(2*data['radius']/g)
    yt.add_field(("gas", "tff"), function=tff, units='Gyr', display_name=r"$t_{\rm ff}$")

    def tcool_tff(field,data):
        return data['tcool']/data['tff']
    yt.add_field(("gas","tcool_tff"),function=tcool_tff, display_name=r"$t_{\rm cool}/t_{\rm ff}$")

    def _metallicity(field, data):
        return data['specific_scalar[0]']*data['density']/(UnitMass/UnitLength**3)
    yt.add_field(("gas","metallicity"), function=_metallicity, units="", display_name=r"$Z/Z_\odot$")

############################################################
####  End  Fielding+17 specific stuff to set the units  ####
############################################################




### Define additional fields for yt
def _my_radial_velocity(field, data):
    xv = data["gas","velocity_x"]
    yv = data["gas","velocity_y"]
    zv = data["gas","velocity_z"]
    center = data.get_field_parameter('center')
    x_hat = data["x"] - center[0]
    y_hat = data["y"] - center[1]
    z_hat = data["z"] - center[2]
    r = np.sqrt(x_hat*x_hat+y_hat*y_hat+z_hat*z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return xv*x_hat + yv*y_hat + zv*z_hat

yt.add_field(("gas","rv"),
             function=_my_radial_velocity,
             units="km/s",
             take_log=False,
             display_name=r"$v_{r}$")

def number_density(field,data):
    return data['density']/(mu*mp)
yt.add_field(("gas","number_density"),function=number_density,units="cm**-3", display_name=r"$n$")

def number_density_H(field,data):
    return data['density']/(muH*mp)
yt.add_field(("gas","number_density_H"),function=number_density_H,units="cm**-3", display_name=r"$n_H$")

def Pkb(field,data):
    return data['pressure']/kb
yt.add_field(("gas","Pkb"),function=Pkb,units="K*cm**-3", display_name=r"$P/k_{\rm B}$")

def Ent(field,data):
    return data['temperature']/((data['density']/(mu*mp))**(2/3.))
yt.add_field(("gas","Ent"),function=Ent,units="K*cm**2", display_name=r"$K$")




### set the units if you need to for your simulation
#units_override = {"length_unit":(UnitLength.value, UnitLength.units),
#                  "time_unit":(UnitTime.value, UnitTime.units),
#                  "mass_unit":(UnitMass.value, UnitMass.units)}

### get all of the data files for your simulation and set the units
#ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)

def get_fn(i):
    a0=''
    if (i<10):
      a0='000'
    if (10<=i<100):
      a0='00'
    if (100<=i<=999):
      a0='0'
    filen='DD'+a0+str(i)+'/sb_'+a0+str(i)
    return filen

METALLICITY=10**-0.5
ii=400
os.system("mkdir profiles_2d")
fn=get_fn(ii)
fn="/simons/scratch/mli/CGM_180821_HSE_1e-6_Trunc0.002kpc_SN2halfSphere_lowz3kpc_SFR3/DD0"+str(ii)+"/sb_0"+str(ii)
print fn
ts = yt.load(fn)
print ("1")
#r200m = YTQuantity(340,'kpc')
H0 = 70 * km /s / Mpc
mu = 0.62
muH = 1/0.7
Tcool =1e4
M15= 1e-3
Tcool *= K
OL = 0.73
Om = 0.27
UnitLength = YTQuantity((G * M15 * 1e15 * Msun / H0**2)**(1./3.)).convert_to_units('kpc')
UnitTime = YTQuantity(1/H0).convert_to_units('yr')
UnitMass = YTQuantity(M15 * 1e15 * Msun).convert_to_units('Msun')
UnitTemp = 4.688*0.62*M15**(2./3.) *keV
#kbTfloor = Tigm*UnitTemp
zz =0.0
H = np.sqrt(OL + (1+zz)**3 * Om)
r200m = UnitLength * (H**2/( (1+zz)**3 * Om))**(1./3.)*(10*H)**(-2./3.)




i_file = rank
while i_file <=1: #len(ts):
    ### for some reason I have to do this anew each time, maybe this bug has been fixed but whatever
#    units_override = {"length_unit":(UnitLength.value, UnitLength.units),
#                      "time_unit":(UnitTime.value, UnitTime.units),
#                      "mass_unit":(UnitMass.value, UnitMass.units)}
#    ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)

    ### select your data file
#    ds = ts[i_file]
    ds = ts
    print ("1")

    ### create a sphere centered on your galaxy
    sphere = ds.sphere([0.,0.,0.], (2.00*r200m.value, "kpc"))
    fields_total =["cell_volume","cell_mass"]
    profile_pressure_entropy = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb", "Ent"],
                                 fields=fields_total,
                                 n_bins=(20,70,65),
                                 units=dict(radius="kpc",Pkb="K*cm**-3",Ent="K*cm**2"),
                                 logs=dict(radius=False,Pkb=True,Ent=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*r200m.value), Pkb=(1e-2,1e5), Ent=(1e4,10**10.5)))

    profile_density_temperature = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "number_density", "temperature"],
                                 fields=fields_total,
                                 n_bins=(20,60,50),
                                 units=dict(radius="kpc",number_density="cm**-3",temperature="K"),
                                 logs=dict(radius=False,number_density=True,temperature=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*r200m.value), number_density=(1e-7,1e-1), temperature=(10**3,10**8)))

    profile_pressure = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb"],
                                 fields=fields_total,
                                 n_bins=(200,70),
                                 units=dict(radius="kpc",Pkb="K*cm**-3"),
                                 logs=dict(radius=True,Pkb=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), Pkb=(1e-2,1e5)))
    profile_entropy = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Ent"],
                                 fields=fields_total,
                                 n_bins=(200,65),
                                 units=dict(radius="kpc",Ent="K*cm**2"),
                                 logs=dict(radius=True,Ent=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), Ent=(1e4,10**10.5)))
    profile_temperature = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "temperature"],
                                 fields=fields_total,
                                 n_bins=(200,50),
                                 units=dict(radius="kpc",temperature="K"),
                                 logs=dict(radius=True,temperature=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), temperature=(10**3,10**8)))
    profile_number_density = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "number_density"],
                                 fields=fields_total,
                                 n_bins=(200,60),
                                 units=dict(radius="kpc",number_density="cm**-3"),
                                 logs=dict(radius=True,number_density=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), number_density=(1e-7,1e-1)))
    profile_radial_velocity = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "velocity_spherical_radius"],
                                 fields=fields_total,
                                 n_bins=(200,150),
                                 units=dict(radius="kpc",velocity_spherical_radius="km/s"),
                                 logs=dict(radius=True,velocity_spherical_radius=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), velocity_spherical_radius=(-500,1000)))
    profile_azimuthal_velocity = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "velocity_spherical_phi"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",velocity_spherical_phi="km/s"),
                                 logs=dict(radius=True,velocity_spherical_phi=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), velocity_spherical_phi=(-500,500)))
    profile_polar_velocity = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "velocity_spherical_theta"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",velocity_spherical_theta="km/s"),
                                 logs=dict(radius=True,velocity_spherical_theta=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), velocity_spherical_theta=(-500,500)))
    profile_specific_angular_momentum_x = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "specific_angular_momentum_x"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",specific_angular_momentum_x="kpc*km/s"),
                                 logs=dict(radius=True,specific_angular_momentum_x=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), specific_angular_momentum_x=(-2.5e5,2.5e5)))
    profile_specific_angular_momentum_y = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "specific_angular_momentum_y"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",specific_angular_momentum_y="kpc*km/s"),
                                 logs=dict(radius=True,specific_angular_momentum_y=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), specific_angular_momentum_y=(-2.5e5,2.5e5)))
    profile_specific_angular_momentum_z = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "specific_angular_momentum_z"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",specific_angular_momentum_z="kpc*km/s"),
                                 logs=dict(radius=True,specific_angular_momentum_z=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), specific_angular_momentum_z=(-2.5e5,2.5e5)))


    if drummond :
        profile_tcool = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool"],
                                 fields=fields_total,
                                 n_bins=(200,60),
                                 units=dict(radius="kpc",tcool="Gyr"),
                                 logs=dict(radius=True,tcool=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), tcool=(1e-4,1e2)))

        np.savez('profiles_2d/profiles_'+str(ii).zfill(4)+'.npz', 
            r200m = r200m.value,
            halo_mass = UnitMass.value * M15,
            redshift = zz,
            metallicity = METALLICITY,
            time = (ds.current_time/Gyr).value,
            r_r200m_phase = (profile_pressure_entropy.x/r200m).value,
            r_r200m_profile = (profile_pressure.x/r200m).value,
            temperature_bins = (profile_temperature.y_bins).value,
            pressure_bins = (profile_pressure.y_bins).value,
            entropy_bins = (profile_entropy.y_bins).value,
            number_density_bins = (profile_number_density.y_bins).value,
            radial_velocity_bins = (profile_radial_velocity.y_bins).value,
            azimuthal_velocity_bins = (profile_azimuthal_velocity.y_bins).value,
            polar_velocity_bins = (profile_polar_velocity.y_bins).value,
            specific_angular_momentum_x_bins = (profile_specific_angular_momentum_x.y_bins).value,
            specific_angular_momentum_y_bins = (profile_specific_angular_momentum_y.y_bins).value,
            specific_angular_momentum_z_bins = (profile_specific_angular_momentum_z.y_bins).value,
            tcool_bins = (profile_tcool.y_bins).value,
            pressure_entropy_Volume = (profile_pressure_entropy['cell_volume'].in_units('kpc**3').value).T,
            pressure_entropy_Mass = (profile_pressure_entropy['cell_mass'].in_units('Msun').value).T ,
            density_temperature_Volume = (profile_density_temperature ['cell_volume'].in_units('kpc**3').value).T,
            density_temperature_Mass = (profile_density_temperature['cell_mass'].in_units('Msun').value).T ,
            temperature_Volume = (profile_temperature['cell_volume'].in_units('kpc**3').value).T,
            temperature_Mass = (profile_temperature['cell_mass'].in_units('Msun').value).T ,
            number_density_Volume = (profile_number_density['cell_volume'].in_units('kpc**3').value).T,
            number_density_Mass = (profile_number_density['cell_mass'].in_units('Msun').value).T ,
            pressure_Volume = (profile_pressure['cell_volume'].in_units('kpc**3').value).T,
            pressure_Mass = (profile_pressure['cell_mass'].in_units('Msun').value).T ,
            entropy_Volume = (profile_entropy['cell_volume'].in_units('kpc**3').value).T,
            entropy_Mass = (profile_entropy['cell_mass'].in_units('Msun').value).T ,
            radial_velocity_Volume = (profile_radial_velocity['cell_volume'].in_units('kpc**3').value).T,
            radial_velocity_Mass = (profile_radial_velocity['cell_mass'].in_units('Msun').value).T ,
            azimuthal_velocity_Volume = (profile_azimuthal_velocity['cell_volume'].in_units('kpc**3').value).T,
            azimuthal_velocity_Mass = (profile_azimuthal_velocity['cell_mass'].in_units('Msun').value).T ,
            polar_velocity_Volume = (profile_polar_velocity['cell_volume'].in_units('kpc**3').value).T,
            polar_velocity_Mass = (profile_polar_velocity['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_x_Volume = (profile_specific_angular_momentum_x['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_x_Mass = (profile_specific_angular_momentum_x['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_y_Volume = (profile_specific_angular_momentum_y['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_y_Mass = (profile_specific_angular_momentum_y['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_z_Volume = (profile_specific_angular_momentum_z['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_z_Mass = (profile_specific_angular_momentum_z['cell_mass'].in_units('Msun').value).T ,
            tcool_Volume = (profile_tcool['cell_volume'].in_units('kpc**3').value).T,
            tcool_Mass = (profile_tcool['cell_mass'].in_units('Msun').value).T 
        )
    else :
        np.savez('profiles_2d/profiles_'+str(ii).zfill(4)+'.npz', 
            r200m = r200m.value,
            halo_mass = UnitMass.value * M15,
            redshift = zz,
            metallicity = METALLICITY,
            time = (ds.current_time/Gyr).value,
            r_r200m_phase = (profile_pressure_entropy.x/r200m).value,
            r_r200m_profile = (profile_pressure.x/r200m).value,
            temperature_bins = (profile_temperature.y_bins).value,
            pressure_bins = (profile_pressure.y_bins).value,
            entropy_bins = (profile_entropy.y_bins).value,
            number_density_bins = (profile_number_density.y_bins).value,
            radial_velocity_bins = (profile_radial_velocity.y_bins).value,
            azimuthal_velocity_bins = (profile_azimuthal_velocity.y_bins).value,
            polar_velocity_bins = (profile_polar_velocity.y_bins).value,
            specific_angular_momentum_x_bins = (profile_specific_angular_momentum_x.y_bins).value,
            specific_angular_momentum_y_bins = (profile_specific_angular_momentum_y.y_bins).value,
            specific_angular_momentum_z_bins = (profile_specific_angular_momentum_z.y_bins).value,
            pressure_entropy_Volume = (profile_pressure_entropy['cell_volume'].in_units('kpc**3').value).T,
            pressure_entropy_Mass = (profile_pressure_entropy['cell_mass'].in_units('Msun').value).T ,
            density_temperature_Volume = (profile_density_temperature ['cell_volume'].in_units('kpc**3').value).T,
            density_temperature_Mass = (profile_density_temperature['cell_mass'].in_units('Msun').value).T ,
            temperature_Volume = (profile_temperature['cell_volume'].in_units('kpc**3').value).T,
            temperature_Mass = (profile_temperature['cell_mass'].in_units('Msun').value).T ,
            number_density_Volume = (profile_number_density['cell_volume'].in_units('kpc**3').value).T,
            number_density_Mass = (profile_number_density['cell_mass'].in_units('Msun').value).T ,
            pressure_Volume = (profile_pressure['cell_volume'].in_units('kpc**3').value).T,
            pressure_Mass = (profile_pressure['cell_mass'].in_units('Msun').value).T ,
            entropy_Volume = (profile_entropy['cell_volume'].in_units('kpc**3').value).T,
            entropy_Mass = (profile_entropy['cell_mass'].in_units('Msun').value).T ,
            radial_velocity_Volume = (profile_radial_velocity['cell_volume'].in_units('kpc**3').value).T,
            radial_velocity_Mass = (profile_radial_velocity['cell_mass'].in_units('Msun').value).T ,
            azimuthal_velocity_Volume = (profile_azimuthal_velocity['cell_volume'].in_units('kpc**3').value).T,
            azimuthal_velocity_Mass = (profile_azimuthal_velocity['cell_mass'].in_units('Msun').value).T ,
            polar_velocity_Volume = (profile_polar_velocity['cell_volume'].in_units('kpc**3').value).T,
            polar_velocity_Mass = (profile_polar_velocity['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_x_Volume = (profile_specific_angular_momentum_x['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_x_Mass = (profile_specific_angular_momentum_x['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_y_Volume = (profile_specific_angular_momentum_y['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_y_Mass = (profile_specific_angular_momentum_y['cell_mass'].in_units('Msun').value).T ,
            specific_angular_momentum_z_Volume = (profile_specific_angular_momentum_z['cell_volume'].in_units('kpc**3').value).T,
            specific_angular_momentum_z_Mass = (profile_specific_angular_momentum_z['cell_mass'].in_units('Msun').value).T )


    for ir in xrange(len(profile_density_temperature.x)):
        plot=plt.pcolormesh(profile_density_temperature.y_bins, profile_density_temperature.z_bins, 
            (profile_density_temperature['cell_volume'].in_units('kpc**3').value).T[...,ir]/np.sum((profile_density_temperature['cell_volume'].in_units('kpc**3').value).T[...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_density_temperature.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_density_temperature.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/number_density_temperature_Volume_r_'+str(ir).zfill(3)+'_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plot=plt.pcolormesh(profile_density_temperature.y_bins, profile_density_temperature.z_bins, 
            profile_density_temperature['cell_mass'].in_units('Msun').T[...,ir]/np.sum(profile_density_temperature['cell_mass'].in_units('Msun').T[...,ir]),
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_density_temperature.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_density_temperature.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/number_density_temperature_Mass_r_'+str(ir).zfill(3)+'_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        
                                                                
    for ir in xrange(len(profile_pressure_entropy.x)):
        plot=plt.pcolormesh(profile_pressure_entropy.y_bins, profile_pressure_entropy.z_bins, 
            (profile_pressure_entropy['cell_volume'].in_units('kpc**3').value).T[...,ir]/np.sum((profile_pressure_entropy['cell_volume'].in_units('kpc**3').value).T[...,ir]),
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_pressure_entropy.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_pressure_entropy.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/pressure_entropy_Volume_r_'+str(ir).zfill(3)+'_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()       

        plot=plt.pcolormesh(profile_pressure_entropy.y_bins, profile_pressure_entropy.z_bins, 
            profile_pressure_entropy['cell_mass'].in_units('Msun').T[...,ir]/np.sum(profile_pressure_entropy['cell_mass'].in_units('Msun').T[...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_pressure_entropy.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_pressure_entropy.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/pressure_entropy_Mass_r_'+str(ir).zfill(3)+'_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        


    plot=plt.pcolormesh(profile_entropy.x_bins/r200m, profile_entropy.y_bins, 
        (profile_entropy['cell_volume'].T/(np.nansum(profile_entropy['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/entropy_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_entropy.x_bins/r200m, profile_entropy.y_bins, 
        (profile_entropy['cell_mass'].T/(np.nansum(profile_entropy['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/entropy_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_pressure.x_bins/r200m, profile_pressure.y_bins, 
        (profile_pressure['cell_volume'].T/(np.nansum(profile_pressure['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/pressure_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_pressure.x_bins/r200m, profile_pressure.y_bins, 
        (profile_pressure['cell_mass'].T/(np.nansum(profile_pressure['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/pressure_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_temperature.x_bins/r200m, profile_temperature.y_bins, 
        (profile_temperature['cell_volume'].T/(np.nansum(profile_temperature['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/temperature_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_temperature.x_bins/r200m, profile_temperature.y_bins, 
        (profile_temperature['cell_mass'].T/(np.nansum(profile_temperature['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/temperature_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_number_density.x_bins/r200m, profile_number_density.y_bins, 
        (profile_number_density['cell_volume'].T/(np.nansum(profile_number_density['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/number_density_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_number_density.x_bins/r200m, profile_number_density.y_bins, 
        (profile_number_density['cell_mass'].T/(np.nansum(profile_number_density['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/number_density_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_radial_velocity.x_bins/r200m, profile_radial_velocity.y_bins, 
        (profile_radial_velocity['cell_volume'].T/(np.nansum(profile_radial_velocity['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/radial_velocity_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_radial_velocity.x_bins/r200m, profile_radial_velocity.y_bins, 
        (profile_radial_velocity['cell_mass'].T/(np.nansum(profile_radial_velocity['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/radial_velocity_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_azimuthal_velocity.x_bins/r200m, profile_azimuthal_velocity.y_bins, 
        (profile_azimuthal_velocity['cell_volume'].T/(np.nansum(profile_azimuthal_velocity['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/azimuthal_velocity_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_azimuthal_velocity.x_bins/r200m, profile_azimuthal_velocity.y_bins, 
        (profile_azimuthal_velocity['cell_mass'].T/(np.nansum(profile_azimuthal_velocity['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_\phi\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/azimuthal_velocity_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_polar_velocity.x_bins/r200m, profile_polar_velocity.y_bins, 
        (profile_polar_velocity['cell_volume'].T/(np.nansum(profile_polar_velocity['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_\theta \,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/polar_velocity_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_polar_velocity.x_bins/r200m, profile_polar_velocity.y_bins, 
        (profile_polar_velocity['cell_mass'].T/(np.nansum(profile_polar_velocity['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_\theta \,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/polar_velocity_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_x.x_bins/r200m, profile_specific_angular_momentum_x.y_bins, 
        (profile_specific_angular_momentum_x['cell_volume'].T/(np.nansum(profile_specific_angular_momentum_x['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_x\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_x_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_x.x_bins/r200m, profile_specific_angular_momentum_x.y_bins, 
        (profile_specific_angular_momentum_x['cell_mass'].T/(np.nansum(profile_specific_angular_momentum_x['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_x\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_x_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_y.x_bins/r200m, profile_specific_angular_momentum_y.y_bins, 
        (profile_specific_angular_momentum_y['cell_volume'].T/(np.nansum(profile_specific_angular_momentum_y['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_y\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_y_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_y.x_bins/r200m, profile_specific_angular_momentum_y.y_bins, 
        (profile_specific_angular_momentum_y['cell_mass'].T/(np.nansum(profile_specific_angular_momentum_y['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_y\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_y_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_z.x_bins/r200m, profile_specific_angular_momentum_z.y_bins, 
        (profile_specific_angular_momentum_z['cell_volume'].T/(np.nansum(profile_specific_angular_momentum_z['cell_volume'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_z\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_z_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_specific_angular_momentum_z.x_bins/r200m, profile_specific_angular_momentum_z.y_bins, 
        (profile_specific_angular_momentum_z['cell_mass'].T/(np.nansum(profile_specific_angular_momentum_z['cell_mass'],axis=1).d +1)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$j_z\,[\mathrm{kpc\,km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/specific_angular_momentum_z_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    if drummond:
        plot=plt.pcolormesh(profile_tcool.x_bins/r200m, profile_tcool.y_bins, 
            (profile_tcool['cell_volume'].T/(np.nansum(profile_tcool['cell_volume'],axis=1).d +1)), 
            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
        plt.xlabel(r'$r/r_{\rm vir}$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/tcool_Volume_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()

        plot=plt.pcolormesh(profile_tcool.x_bins/r200m, profile_tcool.y_bins, 
            (profile_tcool['cell_mass'].T/(np.nansum(profile_tcool['cell_mass'],axis=1).d +1)), 
            norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
        plt.xlabel(r'$r/r_{\rm vir}$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/tcool_Mass_'+str(ii).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()

    i_file += size
