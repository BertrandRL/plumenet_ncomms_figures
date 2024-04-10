import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import glob

import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader


# Read Carbon Mapper plumes dataset
leaks_df=pd.read_excel("../data/Fig4/carbonmapper_ch4_plumelist_2020_2021.xls",sheet_name=1).sort_values("qplume",ascending=False)

lats=leaks_df['plume_lat'].values
lons=leaks_df['plume_lon'].values
rates=leaks_df['qplume'].values

# Read plumes detected by deep learning model

plume_lats=[]
plume_lons=[]
plume_rates=[]
    
plume_ex=glob.glob('../data/Fig4/number*')
for p in plume_ex:
    plume_lats.append(float(p.split('_')[8].split('.')[0]))
    plume_lons.append(float(p.split('_')[7]))
    plume_rates.append(float(p.split('_')[1]))

    
    
# Plot them on map of the US
shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries = reader.records()

shpfilename2 = shpreader.natural_earth('50m', 'cultural', 'admin_1_states_provinces_lines')
reader2 = shpreader.Reader(shpfilename2)
states = reader2.records()

ulLon=-129
lrLon=-64
lrLat=23
ulLat=45

WESN      = [ulLon, lrLon, lrLat, ulLat]
basemap   = cimgt.Stamen('terrain-background')
fig, axes = plt.subplots(figsize=(10,8), subplot_kw={'projection':basemap.crs})
axes.set_extent(WESN, ccrs.Geodetic())

for country in countries:
    if country.attributes['SOVEREIGNT'] == 'United States of America':
        axes.add_geometries(country.geometry, ccrs.PlateCarree(), edgecolor=(0, 0, 0), facecolor=(1, 1, 1), zorder=0)
        for state in states:
            axes.add_geometries(state.geometry, ccrs.PlateCarree(), edgecolor=(0, 0, 0), facecolor=(1, 1, 1), linewidth=0.5, zorder=0)

    else:
        axes.add_geometries(country.geometry, ccrs.PlateCarree(), facecolor=(1, 1, 1))

scatter=axes.scatter(lons,lats, s=rates/5,facecolors='none', edgecolors='r', alpha=0.2, zorder=1, transform=ccrs.PlateCarree()) 
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.2, num=10, color='r') #num=10
legend = axes.legend(handles, labels,  loc="lower right", title="Rates (kg/h)")

axes.scatter(plume_lons,plume_lats, s=np.array(plume_rates)/5,facecolors='none', edgecolors='k', alpha=1, zorder=2, transform=ccrs.PlateCarree()) 

plt.rcParams["figure.figsize"] = (30,120)
plt.savefig("../results/Fig4.pdf")
plt.show()
plt.close()

