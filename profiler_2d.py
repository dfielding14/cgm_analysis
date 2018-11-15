import matplotlib
matplotlib.rc('font', family='sans-serif', size=10)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['figure.figsize'] = 5,4
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from matplotlib import cm as CM
import yt
from yt.units import *
from yt.utilities.physical_constants import *
import matplotlib.colors as colors

import os
os.system("mkdir profiles_2d")

#### If you would like analyze many data outputs 
#### it will be much faster to do it in parallel
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




####  Begin  Fielding+17 specific stuff to set the units

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


####  End  Fielding+17 specific stuff to set the units





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

def _metallicity(field, data):
    return data['specific_scalar[0]']*data['density']/(UnitMass/UnitLength**3)
yt.add_field(("gas","metallicity"), function=_metallicity, units="", display_name=r"$Z/Z_\odot$")

def Pkb(field,data):
    return data['pressure']/kb
yt.add_field(("gas","Pkb"),function=Pkb,units="K*cm**-3", display_name=r"$P/k_{\rm B}$")

def Ent(field,data):
    return data['temperature']/((data['density']/(mu*mp))**(2/3.))
yt.add_field(("gas","Ent"),function=Ent,units="K*cm**2", display_name=r"$K$")




### set the units if you need to for your simulation
units_override = {"length_unit":(UnitLength.value, UnitLength.units),
                  "time_unit":(UnitTime.value, UnitTime.units),
                  "mass_unit":(UnitMass.value, UnitMass.units)}

### get all of the data files for your simulation and set the units
ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)


i_file = rank
while i_file < len(ts):
    ### for some reason I have to do this anew each time, maybe this bug has been fixed but whatever
    units_override = {"length_unit":(UnitLength.value, UnitLength.units),
                      "time_unit":(UnitTime.value, UnitTime.units),
                      "mass_unit":(UnitMass.value, UnitMass.units)}
    ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)

    ### select your data file
    ds = ts[i_file]

    ### create a sphere centered on your galaxy
    sphere = ds.sphere([0.,0.,0.], (2.00*r200m.value, "kpc"))
    fields_total =["cell_volume","cell_mass"]
    profile_pressure_entropy = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb", "Ent"],
                                 fields=fields_total,
                                 n_bins=(20,50,65),
                                 units=dict(radius="kpc",Pkb="K*cm**-3",Ent="K*cm**2"),
                                 logs=dict(radius=False,Pkb=True,Ent=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*r200m.value), Pkb=(1,1e5), Ent=(1e4,10**10.5)))

    profile_density_temperature = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "number_density", "temperature"],
                                 fields=fields_total,
                                 n_bins=(20,70,50),
                                 units=dict(radius="kpc",number_density="cm**-3",temperature="K"),
                                 logs=dict(radius=False,number_density=True,temperature=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*r200m.value), number_density=(1e-7,1), temperature=(10**3,10**8)))


    profile_pressure = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb"],
                                 fields=fields_total,
                                 n_bins=(200,50),
                                 units=dict(radius="kpc",Pkb="K*cm**-3"),
                                 logs=dict(radius=True,Pkb=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), Pkb=(1,1e5)))
    profile_entropy = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Ent"],
                                 fields=fields_total,
                                 n_bins=(200,50),
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
                                 n_bins=(200,70),
                                 units=dict(radius="kpc",number_density="cm**-3"),
                                 logs=dict(radius=True,number_density=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), number_density=(1e-7,1)))
    profile_radial_velocity = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "rv"],
                                 fields=fields_total,
                                 n_bins=(200,100),
                                 units=dict(radius="kpc",rv="km/s"),
                                 logs=dict(radius=True,rv=False),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), rv=(-500,500)))
    profile_tcool = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool"],
                                 fields=fields_total,
                                 n_bins=(200,60),
                                 units=dict(radius="kpc",tcool="Gyr"),
                                 logs=dict(radius=True,tcool=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value), tcool=(1e-4,1e2)))

    np.savez('profiles_2d/profiles_'+str(i_file).zfill(4)+'.npz', 
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
        pressure_entropy_Volume = (profile_pressure_entropy['cell_volume'].in_units('kpc**3').value).T,
        pressure_entropy_Mass = (profile_pressure_entropy['cell_mass'].in_units('Msun').value).T ,
        density_temperature_Volume = (profile_density_temperature ['cell_volume'].in_units('kpc**3').value).T,
        density_temperature_Mass = (profile_density_temperature['cell_mass'].in_units('Msun').value).T ,
        temperature_Volume = (profile_temperature['cell_volume'].in_units('kpc**3').value).T,
        temperature_Mass = (profile_temperature['cell_mass'].in_units('Msun').value).T ,
        number_density_Volume = (profile_number_density['cell_volume'].in_units('kpc**3').value).T,
        number_density_Mass = (profile_number_density['cell_mass'].in_units('Msun').value).T ,
        radial_velocity_Volume = (profile_radial_velocity['cell_volume'].in_units('kpc**3').value).T,
        radial_velocity_Mass = (profile_radial_velocity['cell_mass'].in_units('Msun').value).T ,
        tcool_Volume = (profile_tcool['cell_volume'].in_units('kpc**3').value).T,
        tcool_Mass = (profile_tcool['cell_mass'].in_units('Msun').value).T 
    )

    for ir in xrange(len(profile_density_temperature.x)):
        plot=plt.pcolormesh(profile_density_temperature.y_bins, profile_density_temperature.z_bins, 
            (profile_density_temperature['cell_volume'].in_units('kpc**3').value).T[...,ir]/np.sum((profile_density_temperature['cell_volume'].in_units('kpc**3').value).T[...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$\rho\,[\mathrm{g\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_density_temperature.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_density_temperature.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.savefig('profiles_2d/number_density_temperature_Volume_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plot=plt.pcolormesh(profile_density_temperature.y_bins, profile_density_temperature.z_bins, 
            profile_density_temperature['cell_mass'].in_units('Msun').T[...,ir]/np.sum(profile_density_temperature['cell_mass'].in_units('Msun').T[...,ir]),
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$\rho\,[\mathrm{g\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_density_temperature.x_bins[ir]/r200m,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_density_temperature.x_bins[ir+1]/r200m,1))+r'$',fontsize=10)
        plt.savefig('profiles_2d/number_density_temperature_Mass_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
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
        plt.savefig('profiles_2d/pressure_entropy_Volume_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
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
        plt.savefig('profiles_2d/pressure_entropy_Mass_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        


    plot=plt.pcolormesh(profile_entropy.x_bins/r200m, profile_entropy.y_bins, 
        profile_entropy['cell_volume']/np.sum(profile_entropy['cell_volume']), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/entropy_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_entropy.x_bins/r200m, profile_entropy.y_bins, 
        profile_entropy['cell_mass']/np.sum(profile_entropy['cell_mass']), 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/entropy_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_pressure.x_bins/r200m, profile_pressure.y_bins, 
        (profile_pressure['cell_volume']/np.sum(profile_pressure['cell_volume'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/pressure_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_pressure.x_bins/r200m, profile_pressure.y_bins, 
        (profile_pressure['cell_mass']/np.sum(profile_pressure['cell_mass'])).T, 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/pressure_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_temperature.x_bins/r200m, profile_temperature.y_bins, 
        (profile_temperature['cell_volume']/np.sum(profile_temperature['cell_volume'],axis=0)).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/temperature_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_temperature.x_bins/r200m, profile_temperature.y_bins, 
        (profile_temperature['cell_mass']/np.sum(profile_temperature['cell_mass'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/temperature_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_number_density.x_bins/r200m, profile_number_density.y_bins, 
        (profile_number_density['cell_volume']/np.sum(profile_number_density['cell_volume'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/number_density_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_number_density.x_bins/r200m, profile_number_density.y_bins, 
        (profile_number_density['cell_mass']/np.sum(profile_number_density['cell_mass'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/number_density_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_radial_velocity.x_bins/r200m, profile_radial_velocity.y_bins, 
        (profile_radial_velocity['cell_volume']/np.sum(profile_radial_velocity['cell_volume'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/radial_velocity_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_radial_velocity.x_bins/r200m, profile_radial_velocity.y_bins, 
        (profile_radial_velocity['cell_mass']/np.sum(profile_radial_velocity['cell_mass'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/radial_velocity_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_tcool.x_bins/r200m, profile_tcool.y_bins, 
        (profile_tcool['cell_volume']/np.sum(profile_tcool['cell_volume'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/tcool_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_tcool.x_bins/r200m, profile_tcool.y_bins, 
        (profile_tcool['cell_mass']/np.sum(profile_tcool['cell_mass'])).T, 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    plt.savefig('profiles_2d/tcool_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    i_file += size
