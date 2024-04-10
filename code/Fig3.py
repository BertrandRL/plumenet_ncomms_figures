#Imports

import pandas as pd
import numpy as np
import seaborn
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

##############################
#Plotting parameters

n_bins_paper=10

regplot_args={'robust': False,'x_estimator':np.mean, 'x_bins': n_bins_paper, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws={'color':'blue','s':10,'alpha':0.5}
line_kws={'color':'blue','lw':1}


#Fitting functions
def extent_to_rate(x):
    return (c+b*(1-np.exp(-a*x)))


def rate_to_extent(x):
    return ((-1.0/a)*np.log(1-(x-c)/b))

####################
#Load deep detections centered on Carbon Mapper detections

df_leak_detection_stats=pd.read_pickle('../data/Fig3/leak_detection_stats.zip')  

####################

df_leak_detection_stats['Plume rate (kg/h)']=df_leak_detection_stats['leak_q'].astype(float)
df_leak_detection_stats['Plume rate err (kg/h)']=df_leak_detection_stats['leak_dq'].astype(float)
df_leak_detection_stats['leak_dq norm']=df_leak_detection_stats['Plume rate err (kg/h)']/df_leak_detection_stats['Plume rate (kg/h)']

df_leak_detection_stats['Plume rate+ (kg/h)']=df_leak_detection_stats['Plume rate (kg/h)']+df_leak_detection_stats['Plume rate err (kg/h)']

df_leak_detection_stats['leak_lon']=df_leak_detection_stats['leak_lon'].astype(float)
df_leak_detection_stats['leak_lat']=df_leak_detection_stats['leak_lat'].astype(float)

df_leak_detection_stats['DL_auc']=df_leak_detection_stats['DL_auc'].astype(float)

df_leak_detection_stats['DL_enh']=df_leak_detection_stats['DL_enh'].astype(float)
df_leak_detection_stats['DL_detection_bool']=100*df_leak_detection_stats['DL_detection_bool'].astype(bool).astype(float)
df_leak_detection_stats['MBMP_detection_bool']=100*df_leak_detection_stats['MBMP_detection_bool'].astype(bool).astype(float)


df_leak_detection_stats['MBMP_auc']=df_leak_detection_stats['MBMP_auc'].astype(float)
df_leak_detection_stats['MBMP_wass']=df_leak_detection_stats['MBMP_wass'].astype(float)
df_leak_detection_stats['cloud cover t0']=df_leak_detection_stats['cloud cover t0'].astype(float)

df_leak_detection_stats['S2_date t0']= pd.to_datetime(df_leak_detection_stats['S2_date t0'])

df_leak_detection_stats['DL_wass']=df_leak_detection_stats['DL_wass'].astype(float)
df_leak_detection_stats['cloud cover t1']=df_leak_detection_stats['cloud cover t1'].astype(float)


df_leak_detection_stats['leak_date']= pd.to_datetime(df_leak_detection_stats['leak_date'])
df_leak_detection_stats['S2_date t1']= pd.to_datetime(df_leak_detection_stats['S2_date t1'])
df_leak_detection_stats['time_diff']= df_leak_detection_stats['S2_date t1']-df_leak_detection_stats['leak_date']
df_leak_detection_stats['time_diff_S2']= df_leak_detection_stats['S2_date t1']-df_leak_detection_stats['S2_date t0']


leak_bins=np.logspace(0, 3.0, num=10)
bin_indices=np.digitize(df_leak_detection_stats["Plume rate (kg/h)"].values,leak_bins,right=True)
df_leak_detection_stats["Plume rate bin (kg/h)"]=["%i to %i"%(leak_bins[bin_indices[i]-1],leak_bins[bin_indices[i]]) if bin_indices[i]<len(leak_bins) else "%i +"%(leak_bins[-1]) for i in range(len(bin_indices))]


leak_bins_px=np.logspace(0, 3.0, num=10)*20**2
bin_indices=np.digitize(df_leak_detection_stats['Plume size (sqm)'].values,leak_bins_px,right=True)
df_leak_detection_stats['Plume size bin (sqm)']=["%i to %i"%(leak_bins_px[bin_indices[i]-1],leak_bins_px[bin_indices[i]]) if bin_indices[i]<len(leak_bins_px) else "%i +"%(leak_bins_px[-1]) for i in range(len(bin_indices))]

leak_bins_labels=df_leak_detection_stats['Plume rate bin (kg/h)'].unique()
leak_bins_labels_order=np.argsort([int(label.split(' ')[0]) for label in leak_bins_labels])
leak_bins_labels=leak_bins_labels[leak_bins_labels_order]

pix_bins_labels=df_leak_detection_stats['Plume size bin (sqm)'].unique()
pix_bins_labels_order=np.argsort([int(label.split(' ')[0]) for label in pix_bins_labels])
pix_bins_labels=pix_bins_labels[pix_bins_labels_order]




df_leak_detection_stats['GAO'] = [True if 'GAO' in leak_ID else False for leak_ID in df_leak_detection_stats['candidate_ID'].values]
df_leak_detection_stats['Permian Basin']=[True if (lon>-109.1 and lon <-99.5 and lat>30.0 and lat<37.0) else False for lon,lat in zip(df_leak_detection_stats['leak_lon'].values,df_leak_detection_stats['leak_lat'].values)]

df_leak_detection_stats['West']=[True if (lon<-99.5) else False for lon,lat in zip(df_leak_detection_stats['leak_lon'].values,df_leak_detection_stats['leak_lat'].values)]

################################
# Fit the Carbon Mapper mask size versus the Carbon Mapper leak rate

b=800
c=100#50
x=df_leak_detection_stats['Plume size (sqm)'].values
y=df_leak_detection_stats['Plume rate (kg/h)'].values
a = curve_fit(lambda t,a: c+b*(1-np.exp(-a*t)),  x,  y,p0=np.array([0.00001]))[0]

df_leak_detection_stats["Plume rate* (kg/h)"]=[c+b*(1-np.exp(-a*l)) for l in df_leak_detection_stats['Plume size (sqm)'].values]

regplot_args={'robust': True, 'x_bins': 30, 'logistic': False,'fit_reg':False, 'logx': False}
seaborn.regplot(df_leak_detection_stats,x="Plume size (sqm)", y="Plume rate (kg/h)",**regplot_args,scatter_kws=scatter_kws,line_kws=line_kws)
#plt.plot(np.arange(200000),c+b*(1-np.exp(-a*np.arange(200000))),c='k')
#plt.xlim([0,150000])
#plt.ylim([0,1200])
#plt.xlabel('Plume extent (m$^2$)')
#plt.show()
#plt.close()


################################
# Look at detections in the days preceding a Carbon Mapper detection only

df_leak_detection_filtered=df_leak_detection_stats[(np.abs(stats.zscore(df_leak_detection_stats[['Plume rate (kg/h)','DL_detection_bool']])) < 3.0).all(axis=1)]

N_days_out='7D'
N_days_S2='15D'
max_cloud_cover=0.5

df_leak_detection_filtered=df_leak_detection_filtered[
    (df_leak_detection_stats['time_diff'].abs()< pd.Timedelta(N_days_out)) 
    & (df_leak_detection_stats['cloud cover t0']<max_cloud_cover)
    & (df_leak_detection_stats['cloud cover t1']<max_cloud_cover)
]

df_leak_detection_filtered_GAO=df_leak_detection_filtered[df_leak_detection_filtered['GAO']==True]
df_leak_detection_filtered_AVIRIS=df_leak_detection_filtered[df_leak_detection_filtered['GAO']==False]
df_leak_detection_persistent=df_leak_detection_stats[df_leak_detection_stats['persistence']==True]



################################
#Load detections away from Carbon Mapper detections



df_leak_detection_stats_NOPLUMES=pd.read_csv("../data/Fig3/false_positive_detections.csv")
df_leak_detection_stats_NOPLUMES["Plume rate* (kg/h)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES))
df_leak_detection_stats_NOPLUMES["Plume size (sqm)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES))
df_leak_detection_stats_NOPLUMES["detection"]=100.0*df_leak_detection_stats_NOPLUMES["detection"]
df_leak_detection_stats_NOPLUMES=df_leak_detection_stats_NOPLUMES#[::10]


df_leak_detection_stats_NOPLUMES_NM=pd.read_csv("../data/Fig3/NM_false_positive_detections.csv")
df_leak_detection_stats_NOPLUMES_NM["Plume rate* (kg/h)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES_NM))
df_leak_detection_stats_NOPLUMES_NM["Plume size (sqm)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES_NM))
df_leak_detection_stats_NOPLUMES_NM["detection"]=100.0*df_leak_detection_stats_NOPLUMES_NM["detection"]
df_leak_detection_stats_NOPLUMES_NM=df_leak_detection_stats_NOPLUMES_NM[::10]

df_leak_detection_stats_NOPLUMES_Permian=pd.read_csv("../data/Fig3/Permian_false_positive_detections.csv")
df_leak_detection_stats_NOPLUMES_Permian["Plume rate* (kg/h)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES_Permian))
df_leak_detection_stats_NOPLUMES_Permian["Plume size (sqm)"]=np.zeros(len(df_leak_detection_stats_NOPLUMES_Permian))
df_leak_detection_stats_NOPLUMES_Permian["detection"]=100.0*df_leak_detection_stats_NOPLUMES_Permian["detection"]
df_leak_detection_stats_NOPLUMES_Permian=df_leak_detection_stats_NOPLUMES_Permian[::10]



#########################


regplot_args_NOPLUMES_Permian={'robust': False,'x_estimator':np.mean, 'x_bins': 1, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws_NOPLUMES_Permian={'color':'k','s':10,'alpha':1.0}
line_kws_NOPLUMES_Permian={'color':'k','lw':1}

regplot_args_NOPLUMES_NM={'robust': False,'x_estimator':np.mean, 'x_bins': 1, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws_NOPLUMES_NM={'color':'g','s':10,'alpha':0.5}
line_kws_NOPLUMES_NM={'color':'g','lw':1}

regplot_args_NOPLUMES={'robust': False,'x_estimator':np.mean, 'x_bins': 1, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws_NOPLUMES={'color':'y','s':10,'alpha':0.5}
line_kws_NOPLUMES={'color':'y','lw':1}


regplot_args={'robust': False,'x_estimator':np.mean, 'x_bins': n_bins_paper, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws={'color':'blue','s':10,'alpha':0.5}
line_kws={'color':'blue','lw':1}

regplot_args_GAO={'robust': False,'x_estimator':np.mean, 'x_bins': n_bins_paper, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws_GAO={'color':'teal','s':10,'alpha':0.5}
line_kws_GAO={'color':'teal','lw':1}


regplot_args_MBMP={'robust': False,'x_estimator':np.mean, 'x_bins': n_bins_paper, 'logistic': True,'fit_reg':False, 'logx': False}
scatter_kws_MBMP={'color':'r','s':10,'alpha':0.5}
line_kws_MBMP={'color':'r','lw':1}



with matplotlib.rc_context({"lines.linewidth": 1}):
    fig,ax=plt.subplots()
    seaborn.regplot(df_leak_detection_filtered_AVIRIS, x="Plume size (sqm)", y="DL_detection_bool",**regplot_args,scatter_kws=scatter_kws,line_kws=line_kws,ax=ax,label='AVIRIS-NG')
    seaborn.regplot(df_leak_detection_filtered_GAO, x="Plume size (sqm)", y="DL_detection_bool",**regplot_args_GAO,scatter_kws=scatter_kws_GAO,line_kws=line_kws_GAO,ax=ax,label='GAO')
    seaborn.regplot(df_leak_detection_stats_NOPLUMES_Permian, x="Plume size (sqm)", y="detection",**regplot_args_NOPLUMES_Permian,scatter_kws=scatter_kws_NOPLUMES_Permian,line_kws=line_kws_NOPLUMES_Permian,ax=ax,label='Detection rate near operations')

    seaborn.regplot(df_leak_detection_stats_NOPLUMES_NM, x="Plume size (sqm)", y="detection",**regplot_args_NOPLUMES_NM,scatter_kws=scatter_kws_NOPLUMES_NM,line_kws=line_kws_NOPLUMES_NM,ax=ax,label='False positive rate, USA, NM')
    seaborn.regplot(df_leak_detection_stats_NOPLUMES, x="Plume size (sqm)", y="detection",**regplot_args_NOPLUMES,scatter_kws=scatter_kws_NOPLUMES,line_kws=line_kws_NOPLUMES,ax=ax,label='False positive rate, test set')
    ax.set_ylabel("Deep detection %% \n%i days prior to airborne detection"%int(N_days_out[0]))
    ax.set_xlabel("Plume extent (m$^2$)")

    ax.text(35000, 23, '~ Mean leak persistence in catalogue', ha='left', va='center')
    ax.axhspan(20, 26, facecolor='k', alpha=0.1)  

    seaborn.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset=3, trim=False)


    ax.legend()
    ax.yaxis.grid(True)
    ax.xaxis.grid(False) 
    secax_x2 = ax.secondary_xaxis(
        -0.2, functions=(extent_to_rate, rate_to_extent))
    secax_x2.set_xlabel('Plume rate* (kg/h)')
    plt.tight_layout()
    plt.savefig("../results/Fig3.pdf")
    plt.show()
    plt.close()



