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


radial_velocity_mean_MW = np.zeros(len(profile_radial_velocity.x))*km/s
radial_velocity_std_MW = np.zeros(len(profile_radial_velocity.x))*km/s
for i in xrange(len(profile_radial_velocity.x)):
    if np.sum(profile_radial_velocity['cell_mass'][i])>0:
        radial_velocity_mean_MW[i] = np.average(profile_radial_velocity.y, weights = profile_radial_velocity['cell_mass'][i])
        radial_velocity_std_MW[i] = np.sqrt(np.average( np.square(profile_radial_velocity.y - radial_velocity_mean_MW[i]), weights = profile_radial_velocity['cell_mass'][i]))


azimuthal_velocity_mean_MW = np.zeros(len(profile_azimuthal_velocity.x))*km/s
azimuthal_velocity_std_MW = np.zeros(len(profile_azimuthal_velocity.x))*km/s
for i in xrange(len(profile_azimuthal_velocity.x)):
    if np.sum(profile_azimuthal_velocity['cell_mass'][i])>0:
        azimuthal_velocity_mean_MW[i] = np.average(profile_azimuthal_velocity.y, weights = profile_azimuthal_velocity['cell_mass'][i])
        azimuthal_velocity_std_MW[i] = np.sqrt(np.average( np.square(profile_azimuthal_velocity.y - azimuthal_velocity_mean_MW[i]), weights = profile_azimuthal_velocity['cell_mass'][i]))


polar_velocity_mean_MW = np.zeros(len(profile_polar_velocity.x))*km/s
polar_velocity_std_MW = np.zeros(len(profile_polar_velocity.x))*km/s
for i in xrange(len(profile_polar_velocity.x)):
    if np.sum(profile_polar_velocity['cell_mass'][i])>0:
        polar_velocity_mean_MW[i] = np.average(profile_polar_velocity.y, weights = profile_polar_velocity['cell_mass'][i])
        polar_velocity_std_MW[i] = np.sqrt(np.average( np.square(profile_polar_velocity.y - polar_velocity_mean_MW[i]), weights = profile_polar_velocity['cell_mass'][i]))


full_velocity_std_MW = np.sqrt(radial_velocity_std_MW**2+azimuthal_velocity_std_MW**2+polar_velocity_std_MW**2)

radial_velocity_mean_VW = np.zeros(len(profile_radial_velocity.x))*km/s
radial_velocity_std_VW = np.zeros(len(profile_radial_velocity.x))*km/s
for i in xrange(len(profile_radial_velocity.x)):
    if np.sum(profile_radial_velocity['cell_volume'][i])>0:
        radial_velocity_mean_VW[i] = np.average(profile_radial_velocity.y, weights = profile_radial_velocity['cell_volume'][i])
        radial_velocity_std_VW[i] = np.sqrt(np.average( np.square(profile_radial_velocity.y - radial_velocity_mean_VW[i]), weights = profile_radial_velocity['cell_volume'][i]))


azimuthal_velocity_mean_VW = np.zeros(len(profile_azimuthal_velocity.x))*km/s
azimuthal_velocity_std_VW = np.zeros(len(profile_azimuthal_velocity.x))*km/s
for i in xrange(len(profile_azimuthal_velocity.x)):
    if np.sum(profile_azimuthal_velocity['cell_volume'][i])>0:
        azimuthal_velocity_mean_VW[i] = np.average(profile_azimuthal_velocity.y, weights = profile_azimuthal_velocity['cell_volume'][i])
        azimuthal_velocity_std_VW[i] = np.sqrt(np.average( np.square(profile_azimuthal_velocity.y - azimuthal_velocity_mean_VW[i]), weights = profile_azimuthal_velocity['cell_volume'][i]))


polar_velocity_mean_VW = np.zeros(len(profile_polar_velocity.x))*km/s
polar_velocity_std_VW = np.zeros(len(profile_polar_velocity.x))*km/s
for i in xrange(len(profile_polar_velocity.x)):
    if np.sum(profile_polar_velocity['cell_volume'][i])>0:
        polar_velocity_mean_VW[i] = np.average(profile_polar_velocity.y, weights = profile_polar_velocity['cell_volume'][i])
        polar_velocity_std_VW[i] = np.sqrt(np.average( np.square(profile_polar_velocity.y - polar_velocity_mean_VW[i]), weights = profile_polar_velocity['cell_volume'][i]))


full_velocity_std_VW = np.sqrt(radial_velocity_std_VW**2+azimuthal_velocity_std_VW**2+polar_velocity_std_VW**2)










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
plt.plot(profile_radial_velocity.x/r200m, vr_mean, 'k')
plt.fill_between(profile_radial_velocity.x/r200m, vr_mean+vr_std, vr_mean-vr_std, color='k',alpha=0.5)
plt.plot(profile_all.x/r200m, profile_all['velocity_spherical_radius'].in_units('km/s'), 'k', ls='--')
plt.savefig('test.png', dpi=300)
plt.clf()
























# all_fields=["Pkb","Ent","temperature","number_density","velocity_spherical_radius","velocity_spherical_phi","velocity_spherical_theta"]
# profile_all = yt.create_profile( data_source=sphere,
#                              bin_fields=["radius"],
#                              fields=all_fields,
#                              n_bins=(200),
#                              units=dict(radius="kpc"),
#                              logs=dict(radius=True),
#                              weight_field="cell_mass",
#                              extrema=dict(radius=(0.02*r200m.value,2.0*r200m.value)))
# sphere.set_field_parameter("velocity_spherical_radius_profile", profile_all['velocity_spherical_radius'])
# sphere.set_field_parameter("velocity_spherical_phi_profile", profile_all['velocity_spherical_phi'])
# sphere.set_field_parameter("velocity_spherical_theta_profile", profile_all['velocity_spherical_theta'])
# sphere.set_field_parameter("radii_for_profile", profile_all.x)


# def velocity_spherical_radius_offset_from_mean(field,data):
#     velocity_spherical_radius_profile = data.get_field_parameter("velocity_spherical_radius_profile")
#     radii_for_profile = data.get_field_parameter("radii_for_profile")
#     v_mean = velocity_spherical_radius_profile[np.argmin(np.abs(data['radius']-radii_for_profile))]
#     return np.sqrt(np.square(data['velocity_spherical_radius']-v_mean))
# yt.add_field(("gas","velocity_spherical_radius_offset_from_mean"),function=velocity_spherical_radius_offset_from_mean,units="km/s", display_name=r"$dv_r$")



# def velocity_spherical_theta_dispersion(field,data):
#     return data['velocity_spherical_theta']**2
# yt.add_field(("gas","velocity_spherical_theta_dispersion"),function=velocity_spherical_theta_dispersion,units="km/s", display_name=r"$dv_\theta$")

# def velocity_spherical_radius_dispersion(field,data):
#     return data['velocity_spherical_radius']**2
# yt.add_field(("gas","velocity_spherical_radius_dispersion"),function=velocity_spherical_radius_dispersion,units="km/s", display_name=r"$dv_\phi$")

