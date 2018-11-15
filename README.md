# cgm_analysis
SMAUG CGM working group analysis repo

Included is a basic python script that uses yt to analyze the Fielding+17 CGM simulations. It should be fairly straightforward to adapt this script to work for other yt-compatible simulations.

## Desired analysis products
### Phase plots
In spherical annuli with of widths dr = 0.1 r<sub>200m</sub> ranging between 0 and 2 r<sub>200m</sub> measure the mass and volume weighted pressure-entropy and density-temperature phase distributions.

Bins:

P/kb [K cm<sup>-3</sup>] 	= 1 to 10<sup>0.5</sup>\
K    [K cm<sup>2</sup>]  	= 10<sup>4</sup> to 10<sup>10.5</sup>\
T    [K]       				= 10<sup>3</sup> to 10<sup>8</sup>\
n    [cm<sup>-3</sup>]   	= 10<sup>-7</sup> to 1\
bin widths = 0.1 dex

note that n = ρ / (μ m<sub>p</sub>), where μ≈0.62 for a fully ionized enriched plasma.

### Radial distributions
Measure the mass and volume weighted radial distributions (2d-histogram) of temperature, density, and radial velocity between 0 and 2 r<sub>200m</sub>. 

Bins:
T    [K]       				= 10<sup>3</sup> to 10<sup>8</sup>\
n    [cm<sup>-3</sup>]   	= 10<sup>-7</sup> to 1\
v<sub>r</sub>    [km/s]   	= -500 to 500\
bin widths = 0.1 dex for T and n, and 10 km/s for v<sub>r</sub>\

The radial bins will likely be resolution/simulation dependent, but a good starting point to try is:

r/r<sub>200m</sub> = log<sub>10</sub> 0.02 to log<sub>10</sub> 2

with 200 logarithmically spaced bins, corresponding to ∆log<sub>10</sub> r/r<sub>200m</sub> = 0.01


### Storage formatting
We are going to use `.npz` files for our data storage ([basic usage information](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.savez.html)). Please save the mass-weighted quantities in units of Solar Masses and the volume-weighted quantities in units of kpc<sup>3</sup>.

It is probably easiest to store all of the analysis products for each simulation output in a single `.npz` file. This can be done as follows:

```
np.savez('profiles_my_simulation_name.npz',
	r_r200m_phase = my_1D_array_containing_radial_bins_divided_by_r200m_used_for_phase_diagrams,
	r_r200m_profile = my_1D_array_containing_radial_bins_divided_by_r200m_used_for_radial_profiles,
	temperature_bins = my_temperature_bin_1D_array,
	pressure_bins = my_pressure_bin_1D_array,
	entropy_bins = my_entropy_bin_1D_array,
	number_density_bins = my_number_density_bin_1D_array,
	radial_velocity_bins = my_radial_velocity_bin_1D_array,
	pressure_ent_Volume = my_Phase_distribution_in_each_radial_bin_3D_array,
	pressure_ent_Mass = my_Phase_distribution_in_each_radial_bin_3D_array,
	density_temperature_Volume = my_Phase_distribution_in_each_radial_bin_3D_array,
	density_temperature_Mass = my_Phase_distribution_in_each_radial_bin_3D_array,
	temperature_Volume = my_radial_distribution_2D_array,
	temperature_Mass = my_radial_distribution_2D_array,
	number_density_Volume = my_radial_distribution_2D_array,
	number_density_Mass = my_radial_distribution_2D_array,
	radial_velocity_Volume = my_radial_distribution_2D_array,
	radial_velocity_Mass = my_radial_distribution_2D_array,
	...
)
```

Each user should make their `my_...` using either the provided script or whatever method they want, but it would be great if we could all use the same units and naming convention for the storage to streamline the process down the line, and if people hate my choices please feel free to comment and we can change them. 

### Plotting
To streamline comparisons we should all plot our results in the same way. 
