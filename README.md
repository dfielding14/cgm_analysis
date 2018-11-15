# cgm_analysis
SMAUG CGM working group analysis repo

Included is a basic python script that uses yt to analyze the Fielding+17 CGM simulations. It should be fairly straightforward to adapt this script to work for other yt-compatible simulations.

## Desired analysis products
### Phase plots
In spherical annuli with of widths dr = 0.1 r<sub>200m</sub> ranging between 0 and 2 r<sub>200m</sub> measure the mass and volume weighted pressure-entropy and density-temperature phase distributions.

Bins:

P/kb [K cm^-3] = 1 - 10<sup>0.5</sup>

K    [K cm^2]  = 10<sup>4</sup> - 10<sup>10.5</sup>

T    [K]       = 10<sup>3</sup> - 10<sup>8</sup>

n    [cm^-3]   = 10<sup>-7</sup> - 1 

bin widths = 0.1 dex

### Radial distributions
Measure the mass and volume weighted radial distributions (2d-histogram) of temperature, density, and radial velocity between 0 and 2 r<sub>200m</sub>. 

### Storage formatting
We are going to use `.npz` files for our data storage. 

