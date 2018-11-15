import matplotlib
matplotlib.rc('font', family='sansserif', size=10)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['figure.figsize'] = 4,3
import matplotlib.pyplot as plt
import glob
import numpy as np
import h5py
import re
from scipy import interpolate
from scipy import optimize
from scipy import integrate
from matplotlib import cm as CM
import yt
from yt.units import *
from yt.utilities.physical_constants import *
import matplotlib.colors as colors

import os
os.system("mkdir profiles_2d")

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



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
rvir = UnitLength * (H**2/( (1+zz)**3 * Om))**(1./3.)*(10*H)**(-2./3.)



H_He_Cooling  =np.loadtxt('H_He_cooling.dat')
Tbins         =np.loadtxt('Tbins.dat')
nHbins        =np.loadtxt('nHbins.dat')
Metal_Cooling =np.loadtxt('Metal_cooling.dat')

Metal_Cooling = METALLICITY*Metal_Cooling


f_Metal_Cooling = interpolate.RegularGridInterpolator((np.log10(Tbins), np.log10(nHbins)),Metal_Cooling)
f_H_He_Cooling = interpolate.RegularGridInterpolator( (np.log10(Tbins), np.log10(nHbins)), H_He_Cooling)
f_Cooling = interpolate.RegularGridInterpolator(      (np.log10(Tbins), np.log10(nHbins)),Metal_Cooling+H_He_Cooling, bounds_error=False, fill_value=1e-35)

vf_Cooling = np.vectorize(f_Cooling)

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
    g = G*UnitMass/np.square(data['radius']) * (np.log(1.+c*data['radius']/rvir) - c*data['radius']/rvir/(1.+c*data['radius']/rvir))/(np.log(1.+c) - c/(1.+c))
    return np.sqrt(2*data['radius']/g)
yt.add_field(("gas", "tff"), function=tff, units='Gyr', display_name=r"$t_{\rm ff}$")

def tcool_tff(field,data):
    return data['tcool']/data['tff']
yt.add_field(("gas","tcool_tff"),function=tcool_tff, display_name=r"$t_{\rm cool}/t_{\rm ff}$")


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



def tflow(field,data):
    return data['radius']/data['rv']
yt.add_field(("gas", "tflow"), function=tflow, units='Gyr', display_name=r"$t_{\rm flow}$")


def tcool_tflow(field,data):
    return data['tcool']/data['tflow']
yt.add_field(("gas","tcool_tflow"),function=tcool_tff, display_name=r"$t_{\rm cool}/t_{\rm flow}$")

def tcool_tflow_abs(field,data):
    return np.abs(data['tcool']/data['tflow'])
yt.add_field(("gas","tcool_tflow_abs"),function=tcool_tflow_abs, display_name=r"$t_{\rm cool}/t_{\rm flow}$",force_override=True)

def tcool_tflow_in(field,data):
    return data['tcool']/(-1.0*data['tflow'])
yt.add_field(("gas","tcool_tflow_in"),function=tcool_tflow_in, display_name=r"$t_{\rm cool}/t_{\rm flow}$",force_override=True)


def number_density_H(field,data):
    return data['density']/(muH*mp)
yt.add_field(("gas","number_density_H"),function=number_density_H,units="cm**-3", display_name=r"$n_H$")

def Mach_r(field,data):
    return -1.0*data['rv']/np.sqrt(5./3. * data['pressure']/data['density'])
yt.add_field(("gas","Mach_r"),function=Mach_r,units="", display_name=r"$\mathcal{M}_{r} = v_r/c_s$")


def _metallicity(field, data):
    return data['specific_scalar[0]']*data['density']/(UnitMass/UnitLength**3)
yt.add_field(("gas","metallicity"), function=_metallicity, units="", display_name=r"$Z/Z_\odot$")

def Pkb(field,data):
    return data['pressure']/kb
yt.add_field(("gas","Pkb"),function=Pkb,units="K*cm**-3", display_name=r"$P/k_{\rm B}$")

def Ent(field,data):
    return data['temperature']/((data['density']/(mu*mp))**(2/3.))
yt.add_field(("gas","Ent"),function=Ent,units="K*cm**2", display_name=r"$K$")


units_override = {"length_unit":(UnitLength.value, UnitLength.units),
                  "time_unit":(UnitTime.value, UnitTime.units),
                  "mass_unit":(UnitMass.value, UnitMass.units)}
ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)


f_gal = 0.025

my_storage={}

i_file = rank
while i_file < len(ts):
    units_override = {"length_unit":(UnitLength.value, UnitLength.units),
                      "time_unit":(UnitTime.value, UnitTime.units),
                      "mass_unit":(UnitMass.value, UnitMass.units)}
    ts = yt.load("id0/galaxyhalo.*.vtk", units_override=units_override)
    ds = ts[i_file]
    sphere = ds.sphere([0.,0.,0.], (1.99*rvir.value, "kpc"))
    fields_total =["cell_volume","cell_mass"]
    profile_P_Ent = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb", "Ent"],
                                 fields=fields_total,
                                 n_bins=(20,70,50),
                                 units=dict(radius="kpc",Pkb="K*cm**-3",Ent="K*cm**2"),
                                 logs=dict(radius=False,Pkb=True,Ent=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*rvir.value), Pkb=(1e-2,1e5), Ent=(1e4,1e9)))

    profile_rho_T = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "density", "temperature"],
                                 fields=fields_total,
                                 n_bins=(20,70,50),
                                 units=dict(radius="kpc",density="g*cm**-3",temperature="K"),
                                 logs=dict(radius=False,density=True,temperature=True),
                                 weight_field=None,
                                 extrema=dict(radius=(0,2*rvir.value), density=(1e-31,1e-24), temperature=(10**3.5,10**8.5)))


    profile_P = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Pkb"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,70),
                                 units=dict(radius="kpc",Pkb="K*cm**-3"),
                                 logs=dict(radius=True,Pkb=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), Pkb=(1e-2,1e5)))
    profile_Ent = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Ent"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,50),
                                 units=dict(radius="kpc",Ent="K*cm**2"),
                                 logs=dict(radius=True,Ent=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), Ent=(1e4,1e9)))
    profile_T = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "temperature"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,50),
                                 units=dict(radius="kpc",temperature="K"),
                                 logs=dict(radius=True,temperature=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), temperature=(10**3.5,10**8.5)))
    profile_nH = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "number_density_H"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,60),
                                 units=dict(radius="kpc",number_density_H="cm**-3"),
                                 logs=dict(radius=True,number_density_H=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), number_density_H=(1e-7,1e-1)))
    profile_vr = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "rv"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,60),
                                 units=dict(radius="kpc",rv="km/s"),
                                 logs=dict(radius=True,rv=False),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), rv=(-300,300)))
    profile_tcool = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,50),
                                 units=dict(radius="kpc",tcool="Gyr"),
                                 logs=dict(radius=True,tcool=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), tcool=(1e-4,1e2)))
    profile_Mach_r = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "Mach_r"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,50),
                                 units=dict(radius="kpc",Mach_r=""),
                                 logs=dict(radius=True,Mach_r=False),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), Mach_r=(-5,5)))
    profile_tcool_tff = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool_tff"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,80),
                                 units=dict(radius="kpc",tcool_tff=""),
                                 logs=dict(radius=True,tcool_tff=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), tcool_tff=(1e-4,1e4)))
    profile = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool_tflow_in"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,80),
                                 units=dict(radius="kpc",tcool_tflow_in=""),
                                 logs=dict(radius=True,tcool_tflow_in=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), tcool_tflow_in=(1e-4,1e4)))
    profile_abs = yt.create_profile( data_source=sphere,
                                 bin_fields=["radius", "tcool_tflow_abs"],
                                 fields=fields_total,
                                 n_bins=(60.*nlevels,80),
                                 units=dict(radius="kpc",tcool_tflow_abs=""),
                                 logs=dict(radius=True,tcool_tflow_abs=True),
                                 weight_field=None,
                                 extrema=dict(radius=(1.5*f_gal*rvir.value,1.99*rvir.value), tcool_tflow_abs=(1e-4,1e4)))


    np.savez('profiles_2d/profiles_'+str(i_file).zfill(4)+'.npz', 
                                  r_rvir                 = (profile.x/rvir).value, 
                                  r_kpc                  = (profile.x/kpc).value, 
                                  T_bins                 = (profile_T.y_bins).value,
                                  nH_bins                = (profile_nH.y_bins).value,
                                  tcool_tflow_bins       = (profile.y_bins).value,
                                  tcool_tff_bins         = (profile_tcool_tff.y_bins).value,
                                  tcool_bins             = (profile_tcool.y_bins).value,
                                  Mach_r_bins            = (profile_Mach_r.y_bins).value,
                                  vr_bins                = (profile_vr.y_bins).value,
                                  Ent_bins               = (profile_Ent.y_bins).value,
                                  P_bins                 = (profile_P.y_bins).value,
                                  rho_bins               = (profile_rho_T.y_bins).value,
                                  P_Ent_Volume           = (profile_P_Ent['cell_volume'].in_units('kpc**3').value).T,
                                  P_Ent_Mass             = (profile_P_Ent['cell_mass'].in_units('Msun').value).T ,
                                  rho_T_Volume           = (profile_rho_T ['cell_volume'].in_units('kpc**3').value).T,
                                  rho_T_Mass             = (profile_rho_T['cell_mass'].in_units('Msun').value).T ,
                                  Ent_Volume             = np.array([ profile_Ent['cell_volume'][:,i]/(4*pi*np.diff(profile_Ent.x_bins)*profile_Ent.x**2) for i in xrange(len(profile_Ent.y))]),
                                  Ent_Mass               = (profile_Ent['cell_mass'].in_units('Msun').value).T ,
                                  P_Volume               = np.array([ profile_P['cell_volume'][:,i]/(4*pi*np.diff(profile_P.x_bins)*profile_P.x**2) for i in xrange(len(profile_P.y))]),
                                  P_Mass                 = (profile_P['cell_mass'].in_units('Msun').value).T,
                                  T_Volume               = np.array([ profile_T['cell_volume'][:,i]/(4*pi*np.diff(profile_T.x_bins)*profile_T.x**2) for i in xrange(len(profile_T.y))]),
                                  T_Mass                 = (profile_T['cell_mass'].in_units('Msun').value).T ,
                                  nH_Volume              = np.array([ profile_nH['cell_volume'][:,i]/(4*pi*np.diff(profile_nH.x_bins)*profile_nH.x**2) for i in xrange(len(profile_nH.y))]),
                                  nH_Mass                = (profile_nH['cell_mass'].in_units('Msun').value).T ,
                                  vr_Volume              = np.array([ profile_vr['cell_volume'][:,i]/(4*pi*np.diff(profile_vr.x_bins)*profile_vr.x**2) for i in xrange(len(profile_vr.y))]),
                                  vr_Mass                = (profile_vr['cell_mass'].in_units('Msun').value).T ,
                                  tcool_Volume           = np.array([ profile_tcool['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool.x_bins)*profile_tcool.x**2) for i in xrange(len(profile_tcool.y))]),
                                  tcool_Mass             = (profile_tcool['cell_mass'].in_units('Msun').value).T ,
                                  Mach_r_Volume          = np.array([ profile_Mach_r['cell_volume'][:,i]/(4*pi*np.diff(profile_Mach_r.x_bins)*profile_Mach_r.x**2) for i in xrange(len(profile_Mach_r.y))]),
                                  Mach_r_Mass            = (profile_Mach_r['cell_mass'].in_units('Msun').value).T ,
                                  tcool_tff_Volume       = np.array([ profile_tcool_tff['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool_tff.x_bins)*profile_tcool_tff.x**2) for i in xrange(len(profile_tcool_tff.y))]),
                                  tcool_tff_Mass         = (profile_tcool_tff['cell_mass'].in_units('Msun').value).T ,
                                  tcool_tflow_Volume     = np.array([ profile['cell_volume'][:,i]/(4*pi*np.diff(profile.x_bins)*profile.x**2) for i in xrange(len(profile.y))]),
                                  tcool_tflow_Mass       = (profile['cell_mass'].in_units('Msun').value).T ,
                                  tcool_tflow_abs_Volume = np.array([ profile_abs['cell_volume'][:,i]/(4*pi*np.diff(profile_abs.x_bins)*profile_abs.x**2) for i in xrange(len(profile_abs.y))]),
                                  tcool_tflow_abs_Mass   = (profile_abs['cell_mass'].in_units('Msun').value).T ,
                                  time                   = (ds.current_time/Gyr).d)

    my_storage[float(ds.current_time.in_units('Gyr').d)] = [
        (profile_P_Ent['cell_volume'].in_units('kpc**3').value).T,
        (profile_P_Ent['cell_mass'].in_units('Msun').value).T ,
        (profile_rho_T['cell_volume'].in_units('kpc**3').value).T,
        (profile_rho_T['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_Ent['cell_volume'][:,i]/(4*pi*np.diff(profile_Ent.x_bins)*profile_Ent.x**2) for i in xrange(len(profile_Ent.y))]),
        (profile_Ent['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_P['cell_volume'][:,i]/(4*pi*np.diff(profile_P.x_bins)*profile_P.x**2) for i in xrange(len(profile_P.y))]),
        (profile_P['cell_mass'].in_units('Msun').value).T,
        np.array([ profile_T['cell_volume'][:,i]/(4*pi*np.diff(profile_T.x_bins)*profile_T.x**2) for i in xrange(len(profile_T.y))]),
        (profile_T['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_nH['cell_volume'][:,i]/(4*pi*np.diff(profile_nH.x_bins)*profile_nH.x**2) for i in xrange(len(profile_nH.y))]),
        (profile_nH['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_vr['cell_volume'][:,i]/(4*pi*np.diff(profile_vr.x_bins)*profile_vr.x**2) for i in xrange(len(profile_vr.y))]),
        (profile_vr['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_tcool['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool.x_bins)*profile_tcool.x**2) for i in xrange(len(profile_tcool.y))]),
        (profile_tcool['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_Mach_r['cell_volume'][:,i]/(4*pi*np.diff(profile_Mach_r.x_bins)*profile_Mach_r.x**2) for i in xrange(len(profile_Mach_r.y))]),
        (profile_Mach_r['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_tcool_tff['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool_tff.x_bins)*profile_tcool_tff.x**2) for i in xrange(len(profile_tcool_tff.y))]),
        (profile_tcool_tff['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile['cell_volume'][:,i]/(4*pi*np.diff(profile.x_bins)*profile.x**2) for i in xrange(len(profile.y))]),
        (profile['cell_mass'].in_units('Msun').value).T ,
        np.array([ profile_abs['cell_volume'][:,i]/(4*pi*np.diff(profile_abs.x_bins)*profile_abs.x**2) for i in xrange(len(profile_abs.y))]),
        (profile_abs['cell_mass'].in_units('Msun').value).T ]

    for ir in xrange(len(profile_rho_T.x)):
        plot=plt.pcolormesh(profile_rho_T.y_bins, profile_rho_T.z_bins, (profile_rho_T['cell_volume'].in_units('kpc**3').value).T[...,ir], norm=colors.LogNorm(vmin=1,vmax=1e6), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,[kpc}^{3}]$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$\rho\,[\mathrm{g\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_rho_T.x_bins[ir]/rvir,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_rho_T.x_bins[ir+1]/rvir,1))+r'$',fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(4,3)
        plt.savefig('profiles_2d/rho_T_V_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plot=plt.pcolormesh(profile_rho_T.y_bins, profile_rho_T.z_bins, profile_rho_T['cell_mass'].in_units('Msun').T[...,ir], norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$\rho\,[\mathrm{g\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_rho_T.x_bins[ir]/rvir,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_rho_T.x_bins[ir+1]/rvir,1))+r'$',fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(4,3)
        plt.savefig('profiles_2d/rho_T_M_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        
                                                                
    for ir in xrange(len(profile_P_Ent.x)):
        plot=plt.pcolormesh(profile_P_Ent.y_bins, profile_P_Ent.z_bins, (profile_P_Ent['cell_volume'].in_units('kpc**3').value).T[...,ir], norm=colors.LogNorm(vmin=1,vmax=1e6), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,[kpc}^{3}]$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_P_Ent.x_bins[ir]/rvir,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_P_Ent.x_bins[ir+1]/rvir,1))+r'$',fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(4,3)
        plt.savefig('profiles_2d/P_Ent_V_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()       

        plot=plt.pcolormesh(profile_P_Ent.y_bins, profile_P_Ent.z_bins, profile_P_Ent['cell_mass'].in_units('Msun').T[...,ir], norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr} \quad '+str(np.round(profile_P_Ent.x_bins[ir]/rvir,1))+r'\leq r/r_{\rm vir}\leq'+str(np.round(profile_P_Ent.x_bins[ir+1]/rvir,1))+r'$',fontsize=10)
        fig = plt.gcf()
        fig.set_size_inches(4,3)
        plt.savefig('profiles_2d/P_Ent_M_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        


    plot=plt.pcolormesh(profile_tcool_tff.x_bins/rvir, profile_tcool_tff.y_bins, np.array([ profile_tcool_tff['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool_tff.x_bins)*profile_tcool_tff.x**2) for i in xrange(len(profile_tcool_tff.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$t_{\rm cool}/t_{\rm ff}$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tff_abs_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_tcool_tff.x_bins/rvir, profile_tcool_tff.y_bins, (profile_tcool_tff['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$t_{\rm cool}/t_{\rm ff}$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tff_abs_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()
    plot=plt.pcolormesh(profile.x_bins/rvir, profile.y_bins, np.array([ profile['cell_volume'][:,i]/(4*pi*np.diff(profile.x_bins)*profile.x**2) for i in xrange(len(profile.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$t_{\rm cool}/t_{\rm flow}$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tflow_in_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile.x_bins/rvir, profile.y_bins, (profile['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$t_{\rm cool}/t_{\rm flow}$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tflow_in_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_abs.x_bins/rvir, profile_abs.y_bins, np.array([ profile_abs['cell_volume'][:,i]/(4*pi*np.diff(profile_abs.x_bins)*profile_abs.x**2) for i in xrange(len(profile_abs.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$|t_{\rm cool}/t_{\rm flow}|$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tflow_abs_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_abs.x_bins/rvir, profile_abs.y_bins, (profile_abs['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.axhline(1, color='k', dashes=[2,2], dash_capstyle='round',alpha=0.75)
    plt.ylabel(r'$|t_{\rm cool}/t_{\rm flow}|$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_tflow_abs_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_Ent.x_bins/rvir, profile_Ent.y_bins, np.array([ profile_Ent['cell_volume'][:,i]/(4*pi*np.diff(profile_Ent.x_bins)*profile_Ent.x**2) for i in xrange(len(profile_Ent.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/Ent_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_Ent.x_bins/rvir, profile_Ent.y_bins, (profile_Ent['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/Ent_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_P.x_bins/rvir, profile_P.y_bins, np.array([ profile_P['cell_volume'][:,i]/(4*pi*np.diff(profile_P.x_bins)*profile_P.x**2) for i in xrange(len(profile_P.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/P_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_P.x_bins/rvir, profile_P.y_bins, (profile_P['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/P_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_T.x_bins/rvir, profile_T.y_bins, np.array([ profile_T['cell_volume'][:,i]/(4*pi*np.diff(profile_T.x_bins)*profile_T.x**2) for i in xrange(len(profile_T.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/T_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_T.x_bins/rvir, profile_T.y_bins, (profile_T['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/T_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_nH.x_bins/rvir, profile_nH.y_bins, np.array([ profile_nH['cell_volume'][:,i]/(4*pi*np.diff(profile_nH.x_bins)*profile_nH.x**2) for i in xrange(len(profile_nH.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n_H\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/nH_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_nH.x_bins/rvir, profile_nH.y_bins, (profile_nH['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n_H\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/nH_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_vr.x_bins/rvir, profile_vr.y_bins, np.array([ profile_vr['cell_volume'][:,i]/(4*pi*np.diff(profile_vr.x_bins)*profile_vr.x**2) for i in xrange(len(profile_vr.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/vr_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_vr.x_bins/rvir, profile_vr.y_bins, (profile_vr['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/vr_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_tcool.x_bins/rvir, profile_tcool.y_bins, np.array([ profile_tcool['cell_volume'][:,i]/(4*pi*np.diff(profile_tcool.x_bins)*profile_tcool.x**2) for i in xrange(len(profile_tcool.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_tcool.x_bins/rvir, profile_tcool.y_bins, (profile_tcool['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/tcool_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    plot=plt.pcolormesh(profile_Mach_r.x_bins/rvir, profile_Mach_r.y_bins, np.array([ profile_Mach_r['cell_volume'][:,i]/(4*pi*np.diff(profile_Mach_r.x_bins)*profile_Mach_r.x**2) for i in xrange(len(profile_Mach_r.y))]), norm=colors.LogNorm(vmin=1e-4, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$\Omega/4\pi$',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\mathcal{M}_r$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/Mach_r_V_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(profile_Mach_r.x_bins/rvir, profile_Mach_r.y_bins, (profile_Mach_r['cell_mass'].in_units('Msun').value).T , norm=colors.LogNorm(vmin=3e3,vmax=3e8), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'$M\,[M_\odot]$',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\mathcal{M}_r$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(ds.current_time/Gyr,2))+r'\,\mathrm{Gyr}$')
    fig = plt.gcf()
    fig.set_size_inches(4,3)
    plt.savefig('profiles_2d/Mach_r_M_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    i_file += size



storage=comm.gather(my_storage,root=0)

def merge_dicts(list_of_dicts):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in list_of_dicts:
        result.update(dictionary)
    return result


if rank==0:
    storage = merge_dicts(storage)
    times = np.sort(storage.keys())
    print times

    P_Ent_Volume           = np.array([ storage[t][0] for t in times ])
    P_Ent_Mass             = np.array([ storage[t][1] for t in times ])
    rho_T_Volume           = np.array([ storage[t][2] for t in times ])
    rho_T_Mass             = np.array([ storage[t][3] for t in times ])
    Ent_Volume             = np.array([ storage[t][4] for t in times ])
    Ent_Mass               = np.array([ storage[t][5] for t in times ])
    P_Volume               = np.array([ storage[t][6] for t in times ])
    P_Mass                 = np.array([ storage[t][7] for t in times ])
    T_Volume               = np.array([ storage[t][8] for t in times ])
    T_Mass                 = np.array([ storage[t][9] for t in times ])
    nH_Volume              = np.array([ storage[t][10] for t in times ])
    nH_Mass                = np.array([ storage[t][11] for t in times ])
    vr_Volume              = np.array([ storage[t][12] for t in times ])
    vr_Mass                = np.array([ storage[t][13] for t in times ])
    tcool_Volume           = np.array([ storage[t][14] for t in times ])
    tcool_Mass             = np.array([ storage[t][15] for t in times ])
    Mach_r_Volume          = np.array([ storage[t][16] for t in times ])
    Mach_r_Mass            = np.array([ storage[t][17] for t in times ])
    tcool_tff_Volume       = np.array([ storage[t][18] for t in times ])
    tcool_tff_Mass         = np.array([ storage[t][19] for t in times ])
    tcool_tflow_Volume     = np.array([ storage[t][20] for t in times ])
    tcool_tflow_Mass       = np.array([ storage[t][21] for t in times ])
    tcool_tflow_abs_Volume = np.array([ storage[t][22] for t in times ])
    tcool_tflow_abs_Mass   = np.array([ storage[t][23] for t in times ])

    np.savez('./profiles_2d/All_profiles.npz', times = times ,
        r_rvir                 = (profile.x/rvir).value, 
        r_kpc                  = (profile.x/kpc).value, 
        T_bins                 = (profile_T.y_bins).value,
        nH_bins                = (profile_nH.y_bins).value,
        tcool_tflow_bins       = (profile.y_bins).value,
        tcool_tff_bins         = (profile_tcool_tff.y_bins).value,
        tcool_bins             = (profile_tcool.y_bins).value,
        Mach_r_bins            = (profile_Mach_r.y_bins).value,
        vr_bins                = (profile_vr.y_bins).value,
        Ent_bins               = (profile_Ent.y_bins).value,
        P_bins                 = (profile_P.y_bins).value,
        rho_bins               = (profile_rho_T.y_bins).value,
        P_Ent_Volume           = P_Ent_Volume,
        P_Ent_Mass             = P_Ent_Mass,
        rho_T_Volume           = rho_T_Volume,
        rho_T_Mass             = rho_T_Mass,
        Ent_Volume             = Ent_Volume,
        Ent_Mass               = Ent_Mass,
        P_Volume               = P_Volume,
        P_Mass                 = P_Mass,
        T_Volume               = T_Volume,
        T_Mass                 = T_Mass,
        nH_Volume              = nH_Volume,
        nH_Mass                = nH_Mass,
        vr_Volume              = vr_Volume,
        vr_Mass                = vr_Mass,
        tcool_Volume           = tcool_Volume,
        tcool_Mass             = tcool_Mass,
        Mach_r_Volume          = Mach_r_Volume,
        Mach_r_Mass            = Mach_r_Mass,
        tcool_tff_Volume       = tcool_tff_Volume,
        tcool_tff_Mass         = tcool_tff_Mass,
        tcool_tflow_Volume     = tcool_tflow_Volume,
        tcool_tflow_Mass       = tcool_tflow_Mass,
        tcool_tflow_abs_Volume = tcool_tflow_abs_Volume,
        tcool_tflow_abs_Mass   = tcool_tflow_abs_Mass)