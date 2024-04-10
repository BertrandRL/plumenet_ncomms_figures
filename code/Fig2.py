
import numpy as np
import seaborn

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

SNR_range=[-2.,0.5]

#Load metrics data
SNRs_array,metrics1_DL,metrics1_CH4 = np.load("../data/Fig2/SNR.npy"),np.load("../data/Fig2/F1_DL.npy"),np.load("../data/Fig2/F1_MBMP.npy")


#DL metrics
y_R2SNR=metrics1_DL.copy()
x_R2SNR=np.log10(SNRs_array)
counts, xbins, ybins = np.histogram2d(x=x_R2SNR, y=y_R2SNR,bins=(40, 40),range=[SNR_range,[0.0,1.0]])

R2_median=np.array([np.median(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))]) for i in range(len(xbins)-1)])
R2_q25=np.array([np.quantile(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))],0.25) for i in range(len(xbins)-1)])
R2_q75=np.array([np.quantile(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))],0.75) for i in range(len(xbins)-1)])
R2_mean=np.array([np.mean(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))]) for i in range(len(xbins)-1)])


#MBMP metrics
y_R2SNR=metrics1_CH4.copy()
x_R2SNR=np.log10(SNRs_array)
counts, xbins, ybins = np.histogram2d(x=x_R2SNR, y=y_R2SNR,bins=(40, 40),range=[SNR_range,[0.0,1.0]])

R2_median_CH4=np.array([np.median(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))]) for i in range(len(xbins)-1)])
R2_q25_CH4=np.array([np.quantile(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))],0.25) for i in range(len(xbins)-1)])
R2_q75_CH4=np.array([np.quantile(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))],0.75) for i in range(len(xbins)-1)])
R2_mean_CH4=np.array([np.mean(y_R2SNR[((x_R2SNR>xbins[i]) & (x_R2SNR<=xbins[i+1]))]) for i in range(len(xbins)-1)])



#Load examples data
SNR_examples=np.load("../data/Fig2/examples_metrics.npy")
B12_examples=np.load("../data/Fig2/examples_B12.npy")
RGBs=np.load("../data/Fig2/examples_RGB.npy")
GT_mask_plots=np.load("../data/Fig2/examples_GT.npy")
MBMP_mask_plots=np.load("../data/Fig2/examples_MBMP.npy")
DL_mask_plots=np.load("../data/Fig2/examples_DL.npy")



#Plot metrics and examples

SNR_order=SNR_examples[:, 0].argsort().tolist()

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(10**xbins[1:],R2_median,c='b',alpha=1.0,label='Deep detection')
ax.plot(10**xbins[1:],R2_q25,c='b',alpha=0.2)
ax.plot(10**xbins[1:],R2_q75,c='b',alpha=0.2)
ax.fill_between(10**xbins[1:],R2_median,R2_q25,color='b',alpha=0.05)
ax.fill_between(10**xbins[1:],R2_median,R2_q75,color='b',alpha=0.05)

ax.plot(10**xbins[1:],R2_median_CH4,c='k',alpha=1.0,label='Multi-band multi-pass')
ax.plot(10**xbins[1:],R2_q25_CH4,c='k',alpha=0.2)
ax.plot(10**xbins[1:],R2_q75_CH4,c='k',alpha=0.2)
ax.fill_between(10**xbins[1:],R2_median_CH4,R2_q25_CH4,color='k',alpha=0.05)
ax.fill_between(10**xbins[1:],R2_median_CH4,R2_q75_CH4,color='k',alpha=0.05)


for SNR_example in SNR_examples:
    ax.scatter(SNR_example[0],SNR_example[1],marker='x',c='k',alpha=0.5)
    ax.scatter(SNR_example[0],SNR_example[2],marker='x',c='b',alpha=0.5)
    ax.vlines(SNR_example[0],0,1,ls='--',color='k',alpha=0.1)

ax.set_xscale('log')
ax.set_xlim(10**np.array(SNR_range))
ax.set_ylim([0.0,1.])
ax.set_xlabel('Signal to noise ratio',fontsize=16)
ax.set_ylabel('F1 score',fontsize=16)

plt.legend(fontsize=12,loc='lower right')

ax.grid(True,which="both",c='black',lw=0.1)

seaborn.despine(offset=0)

plt.tight_layout()
plt.savefig("../results/Fig2A.pdf")
plt.show()
plt.close()

for i in SNR_order:
    SNR_example=SNR_examples[i]
    B12_example=B12_examples[i]
    RGB=RGBs[i]
    GT_mask_plot=GT_mask_plots[i]
    MBMP_mask_plot=MBMP_mask_plots[i]
    DL_mask_plot=DL_mask_plots[i]
    
    f,ax=plt.subplots(4,1,dpi=150,figsize=(2.5,10))
    im0=ax[0].imshow(B12_example,cmap = cm.inferno)
    ax[1].imshow(RGB)
    im1=ax[1].imshow(MBMP_mask_plot,cmap = cm.inferno_r,vmin=0,vmax=1)
    ax[2].imshow(RGB)
    im2=ax[2].imshow(DL_mask_plot,cmap = cm.inferno_r,vmin=0,vmax=1)
    ax[3].imshow(RGB)
    im3=ax[3].imshow(GT_mask_plot,cmap = cm.inferno,vmin=0,vmax=1)

    ax[0].set_title("B12")
    ax[1].set_title("MBMP")
    ax[2].set_title("DL")
    ax[3].set_title("Ground truth")
    
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')

    plt.tight_layout()
    plt.savefig("../results/Fig2B_%i.png"%i)
    plt.show()
    plt.close()  
    

    


