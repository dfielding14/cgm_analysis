# cgm_analysis
SMAUG CGM working group analysis repo

Included is a basic python script that uses yt to analyze the Fielding+17 CGM simulations. It should be fairly straightforward to adapt this script to work for other yt-compatible simulations.

## Desired analysis products
### Phase plots
In spherical annuli with of widths dr = 0.1 r<sub>200m</sub> ranging between 0 and 2 r<sub>200m</sub> measure the mass and volume weighted pressure-entropy and density-temperature phase distributions.

Bins:

P/kb [K cm<sup>-3</sup>] 	= 1 to 10<sup>0.5</sup>    
K    [K cm<sup>2</sup>]  	= 10<sup>4</sup> to 10<sup>10.5</sup>    
T    [K]       				= 10<sup>3</sup> to 10<sup>8</sup>     
n    [cm<sup>-3</sup>]   	= 10<sup>-7</sup> to 1     
bin widths = 0.1 dex

### Radial distributions
Measure the mass and volume weighted radial distributions (2d-histogram) of temperature, density, and radial velocity between 0 and 2 r<sub>200m</sub>. 

Bins:
T    [K]       				= 10<sup>3</sup> to 10<sup>8</sup>    
n    [cm<sup>-3</sup>]   	= 10<sup>-7</sup> to 1    
v<sub>r</sub>    [km/s]   	= -500 to 500     
bin widths = 0.1 dex for T and n, and 10 km/s for v<sub>r</sub>.   

The radial bins will likely be resolution/simulation dependent, but a good starting point to try is:

r/r<sub>200m</sub> = log<sub>10</sub> 0.02 to log<sub>10</sub>2

with 200 logarithmically spaced bins, corresponding to ∆ log<sub>10</sub> r/r<sub>200m</sub> = 0.01


### Storage formatting
We are going to use `.npz` files for our data storage. 

