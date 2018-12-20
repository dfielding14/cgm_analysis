import matplotlib
matplotlib.use('Agg')
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
import glob

files =  np.sort(glob.glob('profiles_2d/*npz'))

for i_file in xrange(len(files)):
    a = np.load(files[i_file])


    for ir in xrange(len(a['r_r200m_phase'])):
        ##Histograms
        # I am not sure it is better to do lines or bars, I will decide once i overplot everything
        # plt.loglog(a['number_density_bins'][:-1]+np.diff(a['number_density_bins'])/2., np.sum(a['density_temperature_Volume'][...,ir],axis=0)/np.sum(a['density_temperature_Volume'][...,ir]))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        # plt.ylabel(r'$\mathrm{Volume\,Fraction}$')
        # plt.ylim((1e-5,1))
        # plt.savefig('profiles_2d/number_density_Volume_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        # plt.clf()   

        plt.bar(a['number_density_bins'][:-1], np.sum(a['density_temperature_Volume'][...,ir],axis=0)/np.sum(a['density_temperature_Volume'][...,ir]), width=0.8*np.diff(a['number_density_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.ylabel(r'$\mathrm{Volume\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/number_density_Volume_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['temperature_bins'][:-1], np.sum(a['density_temperature_Volume'][...,ir],axis=1)/np.sum(a['density_temperature_Volume'][...,ir]), width=0.8*np.diff(a['temperature_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$T\,[\mathrm{K}]$')
        plt.ylabel(r'$\mathrm{Volume\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/temperature_Volume_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['number_density_bins'][:-1], np.sum(a['density_temperature_Mass'][...,ir],axis=0)/np.sum(a['density_temperature_Mass'][...,ir]), width=0.8*np.diff(a['number_density_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.ylabel(r'$\mathrm{Mass\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/number_density_Mass_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['temperature_bins'][:-1], np.sum(a['density_temperature_Mass'][...,ir],axis=1)/np.sum(a['density_temperature_Mass'][...,ir]), width=0.8*np.diff(a['temperature_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$T\,[\mathrm{K}]$')
        plt.ylabel(r'$\mathrm{Mass\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/temperature_Mass_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf() 

        plt.bar(a['pressure_bins'][:-1], np.sum(a['pressure_entropy_Volume'][...,ir],axis=0)/np.sum(a['pressure_entropy_Volume'][...,ir]), width=0.8*np.diff(a['pressure_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.ylabel(r'$\mathrm{Volume\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/pressure_Volume_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['entropy_bins'][:-1], np.sum(a['pressure_entropy_Volume'][...,ir],axis=1)/np.sum(a['pressure_entropy_Volume'][...,ir]), width=0.8*np.diff(a['entropy_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.ylabel(r'$\mathrm{Volume\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/entropy_Volume_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['pressure_bins'][:-1], np.sum(a['pressure_entropy_Mass'][...,ir],axis=0)/np.sum(a['pressure_entropy_Mass'][...,ir]), width=0.8*np.diff(a['pressure_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.ylabel(r'$\mathrm{Mass\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/pressure_Mass_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plt.bar(a['entropy_bins'][:-1], np.sum(a['pressure_entropy_Mass'][...,ir],axis=1)/np.sum(a['pressure_entropy_Mass'][...,ir]), width=0.8*np.diff(a['entropy_bins']), color='none', edgecolor='k',log=True)
        plt.xscale('log')
        plt.xlabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.ylabel(r'$\mathrm{Mass\,Fraction}$')
        plt.ylim((1e-5,1))
        plt.savefig('profiles_2d/entropy_Mass_hist_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plot=plt.pcolormesh(a['number_density_bins'], a['temperature_bins'], 
            a['density_temperature_Volume'][...,ir]/np.sum(a['density_temperature_Volume'][...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((a['r_r200m_phase']-np.diff(a['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( (a['r_r200m_phase']+np.diff(a['r_r200m_phase'])[0]/2.)[ir] )+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/number_density_temperature_Volume_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()   

        plot=plt.pcolormesh(a['number_density_bins'], a['temperature_bins'], 
            a['density_temperature_Mass'][...,ir]/np.sum(a['density_temperature_Mass'][...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$T\,[\mathrm{K}]$')
        plt.xlabel(r'$n\,[\mathrm{cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((a['r_r200m_phase']-np.diff(a['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( (a['r_r200m_phase']+np.diff(a['r_r200m_phase'])[0]/2.)[ir] )+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/number_density_temperature_Mass_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        
                                                                
    for ir in xrange(len(a['r_r200m_phase'])):
        plot=plt.pcolormesh(a['pressure_bins'], a['entropy_bins'], 
            a['pressure_entropy_Volume'][...,ir]/np.sum(a['pressure_entropy_Volume'][...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='plasma')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Volume\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((a['r_r200m_phase']-np.diff(a['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( (a['r_r200m_phase']+np.diff(a['r_r200m_phase'])[0]/2.)[ir] )+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/pressure_entropy_Volume_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()       

        plot=plt.pcolormesh(a['pressure_bins'], a['entropy_bins'], 
            a['pressure_entropy_Mass'][...,ir]/np.sum(a['pressure_entropy_Mass'][...,ir]), 
            norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
        cb = plt.colorbar(plot)
        cb.set_label(r'$\mathrm{Mass\,Fraction}$',rotation=270,fontsize=12,labelpad=15)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
        plt.xlabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
        plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr} \quad '+str( np.round((a['r_r200m_phase']-np.diff(a['r_r200m_phase'])[0]/2.)[ir],1) )+r'\leq r/r_{\rm vir}\leq'+str( (a['r_r200m_phase']+np.diff(a['r_r200m_phase'])[0]/2.)[ir] )+r'$',fontsize=10)
        plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
        plt.savefig('profiles_2d/pressure_entropy_Mass_r_'+str(ir).zfill(3)+'_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
        plt.clf()        


    plot=plt.pcolormesh(a['r_r200m_profile'], a['temperature_bins'], 
        (a['temperature_Volume']/np.sum(a['temperature_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/temperature_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['temperature_bins'], 
        (a['temperature_Mass']/np.sum(a['temperature_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$T\,[\mathrm{K}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/temperature_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['number_density_bins'], 
        (a['number_density_Volume']/np.sum(a['number_density_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/number_density_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['number_density_bins'], 
        (a['number_density_Mass']/np.sum(a['number_density_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$n\,[\mathrm{cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/number_density_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['radial_velocity_bins'], 
        (a['radial_velocity_Volume']/np.sum(a['radial_velocity_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/radial_velocity_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['radial_velocity_bins'], 
        (a['radial_velocity_Mass']/np.sum(a['radial_velocity_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$v_r\,[\mathrm{km/s}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/radial_velocity_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['tcool_bins'], 
        (a['tcool_Volume']/np.sum(a['tcool_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/tcool_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['tcool_bins'], 
        (a['tcool_Mass']/np.sum(a['tcool_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$t_{\rm cool}\,[\mathrm{Gyr}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/tcool_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['entropy_bins'], 
        (a['entropy_Volume']/np.sum(a['entropy_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/entropy_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['entropy_bins'], 
        (a['entropy_Mass']/np.sum(a['entropy_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$K\,[\mathrm{K\,cm}^{2}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/entropy_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['pressure_bins'], 
        (a['pressure_Volume']/np.sum(a['pressure_Volume'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6, vmax=1), cmap='plasma')
    cb = plt.colorbar(plot)
    cb.set_label(r'Volume Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/pressure_Volume_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()

    plot=plt.pcolormesh(a['r_r200m_profile'], a['pressure_bins'], 
        (a['pressure_Mass']/np.sum(a['pressure_Mass'],axis=0)), 
        norm=colors.LogNorm(vmin=1e-6,vmax=1), cmap='viridis')
    cb = plt.colorbar(plot)
    cb.set_label(r'Mass Fraction',rotation=270,fontsize=12,labelpad=15)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$P\,[\mathrm{K\,cm}^{-3}]$')
    plt.xlabel(r'$r/r_{\rm vir}$')
    plt.title(r'$t='+str(np.round(a['time'],2))+r'\,\mathrm{Gyr}$')
    plt.grid(color='grey',linestyle=':', alpha=0.5, linewidth=1.0)
    plt.savefig('profiles_2d/pressure_Mass_'+str(i_file).zfill(4)+'.png',bbox_inches='tight',dpi=200)
    plt.clf()


    i_file += size
