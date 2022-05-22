import numpy as np
import pandas as pd
import h5py 
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('../.matplotlib/matplotlibrc.bin')



primary = "gamma_100TeV"
size = 160

if (primary=="gamma"):
    filename_train = "/lstore/lattes/borjasg/corsika/gamma/photon_alt5200m_qgsii_fluka_N33300.h5"
    particle = "Gamma"
elif (primary=="proton"):
    filename_train = "/lstore/lattes/borjasg/corsika/proton/proton_alt5200m_qgsii_fluka_N29986.h5"
    particle = "Proton"
elif (primary=="iron"):
    filename_train = "/lstore/lattes/borjasg/corsika/iron/iron_alt5200m_qgsii_fluka_N29974.h5"
    particle = "Iron"
elif (primary=="gamma_100TeV"):
    filename_train = "/lstore/lattes/borjasg/exported_data_corsika/patterns_cristina/dataphoton_1TeV_560m_alt5200m_qgsii_fluka_EnergyGround_test_N2137.h5"
    particle = "Gamma"
    size = 560



###########
#Read data#
###########

f = h5py.File(filename_train, mode = "r")

#Read input data for algorithm
group = f["data"]

#Read shower parameters
InfoDF = pd.read_hdf(filename_train, key = "info")
ID_showers = InfoDF.iloc[:,0].values
E0_train = InfoDF.iloc[:,1].values
theta0_train = InfoDF.iloc[:,2].values
Nmuons_train = InfoDF.iloc[:,3].values

#Read input for the algorithm
#X_train: channel1 -> e.m. energy, channel2 -> particles, channel3 -> muons
X_train = group[()]
Y_train = InfoDF.iloc[:,-1].values

#Close file
f.close()





index_to_use = np.where(E0_train>90000)[0][0]
label = "$E_{em}$ [GeV]"


index_to_use = np.where(E0_train>50000)[0][1]
#Plot to see matrix with colors
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
#plt.imshow(X_train[index_to_use,:,:,0], alpha=0.8, cmap='Reds',norm=mpl.colors.LogNorm())
plt.imshow(X_train[index_to_use,:,:], alpha=0.8, cmap='Reds',norm=mpl.colors.LogNorm())
cbar = plt.colorbar()
cbar.ax.set_ylabel(label, rotation=90)
plt.autoscale(enable=True)
plt.title(str(particle)+" shower. $E_0 =$ "+str(np.round((10**-3)*E0_train[index_to_use],2))+" TeV")
plt.xlabel("x [m]")
plt.xticks(ticks = np.linspace(0,X_train.shape[1]-1,8), labels=np.linspace(-size,size,8).astype(int))
plt.ylabel('y [m]')
plt.yticks(ticks = np.linspace(0,X_train.shape[2]-1,8), labels=-1*np.linspace(-size,size,8).astype(int))
plt.locator_params(axis='y', nbins=8)
plt.locator_params(axis='x', nbins=8)
fig.tight_layout()
filename = './figs/'+str(primary)+'_id'+str(index_to_use)+'.pdf'
plt.savefig(filename)
plt.close()
