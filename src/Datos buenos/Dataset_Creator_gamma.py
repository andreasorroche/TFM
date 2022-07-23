#!/usr/bin/env python


"""
The input variables are the following:

 - Lower energy [TeV]
 - Upper energy [TeV]

Example to use the program:

nohup python3 Dataset_Creator_gamma.py 40 60 & 

"""


import numpy as np
import pandas as pd
import h5py 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import copy

    
    

if (len(sys.argv) < 3):
    print("ERROR. No arguments were given to identify the run", file = sys.stderr)
    print("Please indicate the primary for the experiment, the event id to read and output path")
    sys.exit(1)



#Read input variables
lower_limit = int(sys.argv[1])*10**3 #GeV
upper_limit = int(sys.argv[2])*10**3 #GeV

print("Save events in the energy range: [",sys.argv[1],";",sys.argv[2],"] TeV")

################################################################

####################
##Python Functions##
####################


def plot_histogram_2vars(data1,data2,plotname,xlabel_title,name1,name2,n_bins=60,pos="right",density_option=True):
    #Set limits for histograms
    lim_inf = np.min([np.min(data1),np.min(data2)])
    lim_inf = lim_inf-0.45*lim_inf
    lim_sup=np.max([np.max(data1),np.max(data2)])
    lim_sup = lim_sup+0.1*lim_sup
    #Compute histograms
    kwargs = dict(range=([lim_inf,lim_sup]), bins = n_bins, density=density_option)
    n1,x,_ = plt.hist(data1, **kwargs)
    n2,x,_ = plt.hist(data2, **kwargs)
    #Plot figure
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
    plt.step(x[:-1],n1,color="black",where="post", label = name1)
    plt.step(x[:-1],n2,color="red",where="post", label = name2)
    plt.autoscale(enable=True)
    plt.xlabel(xlabel_title, fontsize=18)
    if density_option==True:
        plt.ylabel('Frecuencia relativa', fontsize=18)
    else:     
        plt.ylabel('Frecuencia', fontsize=18)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    #Add box with information:
    textstr = '\n'.join((
    name1,
    'Entradas = %2d' % (len(data1), ),
    'Media = %1.1E' % (np.nanmean(data1), ),
    'Mediana = %1.1E' % (np.nanmedian(data1), ),
    'Desv Est = %1.1E' % (np.nanstd(data1), )))
    textstr2 = '\n'.join((
    name2,
    'Entradas = %2d' % (len(data2), ),
    'Media = %1.1E' % (np.nanmean(data2), ),
    'Mediana = %1.1E' % (np.nanmedian(data2), ),
    'Desv Est = %1.1E' % (np.nanstd(data2), )))
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    if pos == "right":
        ax.text(x=0.70, y=0.78, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.70, y=0.56, s=textstr2, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    elif pos == "left":
        ax.text(x=0.02, y=0.78, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.02, y=0.56, s=textstr2, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    #Set legend and labels
    if lim_sup<500:
        ax.set_xlim((0,150))
    plt.locator_params(axis="x", nbins=7)
    ax.legend(loc = 'best', edgecolor="black",fontsize=20)
    fig.tight_layout()
    plt.savefig(plotname)
    plt.close()


def plot_histogram_3vars(data1,data2,data3,plotname,xlabel_title,name1,name2,name3,n_bins=90,pos="right",density_option=True):
    #Set limits for histograms
    lim_inf = np.min([np.min(data1),np.min(data2),np.min(data3)])
    lim_inf = lim_inf-0.5*lim_inf
    lim_sup=np.max([np.max(data1),np.max(data2),np.max(data3)])
    lim_sup = lim_sup+0.05*lim_sup
    #Compute histograms
    kwargs = dict(range=([lim_inf,lim_sup]), bins = n_bins, density=density_option)
    n1,x,_ = plt.hist(data1, **kwargs)
    n2,x,_ = plt.hist(data2, **kwargs)
    n3,x,_ = plt.hist(data3, **kwargs)
    #Plot figure
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
    plt.step(x[:-1],n1,color="red",where="post", label = name1)
    plt.step(x[:-1],n2,color="black",where="post", label = name2)
    plt.step(x[:-1],n3,color="blue",where="post", label = name3)
    plt.autoscale(enable=True)
    plt.xlabel(xlabel_title, fontsize=18)
    if density_option==True:
        plt.ylabel('Frecuencia relativa', fontsize=18)
    else:     
        plt.ylabel('Frecuencia', fontsize=18)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #Add box with information:
    textstr = '\n'.join((
    name1,
    'Entradas = %2d' % (len(data1), ),
    'Media = %4.2f' % (np.nanmean(data1), ),
    'Mediana = %4.2f' % (np.nanmedian(data1), ),
    'Desv Est = %4.2f' % (np.nanstd(data1), )))
    textstr2 = '\n'.join((
    name2,
    'Entradas = %2d' % (len(data2), ),
    'Media = %4.2f' % (np.nanmean(data2), ),
    'Mediana = %4.2f' % (np.nanmedian(data2), ),
    'Desv Est = %4.2f' % (np.nanstd(data2), )))
    textstr3 = '\n'.join((
    name3,
    'Entradas = %2d' % (len(data3), ),
    'Media = %4.2f' % (np.nanmean(data3), ),
    'Mediana = %4.2f' % (np.nanmedian(data3), ),
    'Desv Est = %4.2f' % (np.nanstd(data3), )))
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    if pos == "right":
        ax.text(x=0.72, y=0.71, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.72, y=0.49, s=textstr2, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.72, y=0.27, s=textstr3, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    elif pos == "left":
        ax.text(x=0.02, y=0.71, s=textstr, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.02, y=0.49, s=textstr2, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
        ax.text(x=0.02, y=0.27, s=textstr3, transform=ax.transAxes, fontsize=14, 
        verticalalignment='top', bbox=props)
    #Set legend and labels
    ax.set_xlim((lim_inf,lim_sup))
    plt.locator_params(axis="x", nbins=7)
    ax.legend(loc = 'best', edgecolor="black",fontsize=20)
    fig.tight_layout()
    plt.savefig(plotname)
    plt.close()


################################################################






#############
#Read Gamma#
#############

filename_gamma = "C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/photon_alt5200m_qgsii_fluka_r560m_3PMTs_fixed.h5"
print("Reading file: ",filename_gamma)


#Read signal at the ground
f_gamma = h5py.File(filename_gamma, mode = "r")
#Read input data for algorithm
group_gamma = f_gamma["data"]
#Read input for the algorithm
X_gamma = group_gamma[()] #X_train: Estimated signal at the ground
#Close file
f_gamma.close()

#Read shower parameters stored in the pandas dataframe: energy
InfoDF_gamma = pd.read_hdf(filename_gamma, key = "info")
E0_gamma = InfoDF_gamma.iloc[:,1].values


#Get index of the events in the required energy range
index_E_gamma = (E0_gamma>=lower_limit)*(E0_gamma<=upper_limit)


#Compute total signal at the ground for each event (sum of all matrix values)
S_total_gamma_AllEvents = X_gamma.sum(axis=(1,2))

S_total_gamma = S_total_gamma_AllEvents[index_E_gamma]

#Apply cut on the signal: S \in [S_mean-S_std;S_mean+S_std]
S_total_mean = np.nanmean(S_total_gamma)
S_total_std = np.nanstd(S_total_gamma)


#Get index of Gamma events around a sigma from the mean value of signal for this energy range
index_cut_Signal_gamma = (S_total_gamma_AllEvents>=S_total_mean-S_total_std)*(S_total_gamma_AllEvents<=S_total_mean+S_total_std)


#Keep these events
X_gamma_cut = X_gamma[index_cut_Signal_gamma]
InfoDF_gamma_cut = InfoDF_gamma.loc[index_cut_Signal_gamma]


E0_gamma_selected = E0_gamma[index_cut_Signal_gamma]


#Delete the old data set
del X_gamma,InfoDF_gamma

#Check number of events within this range 
Nevents_gamma = len(X_gamma_cut)

#Save new data set if there are events:
output_folder = "C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos"
outputfilename_gamma = output_folder+"photon_alt5200m_qgsii_fluka_r560m_3PMTs_"+sys.argv[1]+"-"+sys.argv[2]+"TeV_N"+str(Nevents_gamma)+".h5"


if Nevents_gamma>0:
    print("Saving Gamma data data set: ",outputfilename_gamma)
    #Save file with all the data
    with h5py.File(outputfilename_gamma, 'w') as hf:
                    hf.create_dataset("data", data=X_gamma_cut, compression="gzip", compression_opts=9)
    #Save pandas table with events variables:
    InfoDF_gamma_cut.to_hdf(outputfilename_gamma, key='info', index=False,mode='a')
    print("Events signal Gamma: ",Nevents_gamma)
    print("Events InfoDF Gamma: ",len(InfoDF_gamma_cut))
    

del X_gamma_cut, InfoDF_gamma_cut




#############
#Read proton#
#############


filename_proton = "C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/proton_alt5200m_qgsii_fluka_r560m_3PMTs_fixed.h5"
print("Reading proton data: ",filename_proton)

#Read signal at the ground
f_proton = h5py.File(filename_proton, mode = "r")
#Read input data for algorithm
group_proton = f_proton["data"]
#Read input for the algorithm
X_proton = group_proton[()] #X_train: Estimated signal at the ground
#Close file
f_proton.close()


#Read shower parameters stored in the pandas dataframe
InfoDF_proton = pd.read_hdf(filename_proton, key = "info")
E0_proton = InfoDF_proton.iloc[:,1].values

#Compute total signal at the ground for each event (sum of all matrix values)
S_total_proton = X_proton.sum(axis=(1,2))


#Apply the same cut on the signal that was applied for protons: S \in [S_mean-S_std;S_mean+S_std]
index_cut_Signal_proton = (S_total_proton>=S_total_mean-S_total_std)*(S_total_proton<=S_total_mean+S_total_std)



#Keep these events
E0_proton_selected = E0_proton[index_cut_Signal_proton]
X_proton_cut = X_proton[index_cut_Signal_proton]
InfoDF_proton_cut = InfoDF_proton.loc[index_cut_Signal_proton]

#Delete the old data set
del X_proton,InfoDF_proton

#Check number of events within this range 
Nevents_proton = len(X_proton_cut)

#Save new data set if there are events:
outputfilename_proton = output_folder+"proton_alt5200m_qgsii_fluka_r560m_3PMTs_"+sys.argv[1]+"-"+sys.argv[2]+"TeV-GammaERange_N"+str(Nevents_proton)+".h5"


if Nevents_proton>0:
    print("Saving proton data data set: ",outputfilename_proton)
    #Save file with all the data
    with h5py.File(outputfilename_proton, 'w') as hf:
                    hf.create_dataset("data", data=X_proton_cut, compression="gzip", compression_opts=9)
    #Save pandas table with events variables:
    InfoDF_proton_cut.to_hdf(outputfilename_proton, key='info', index=False,mode='a')
    print("Events signal proton: ",Nevents_proton)
    print("Events InfoDF proton: ",len(InfoDF_proton_cut))
    

del X_proton_cut, InfoDF_proton_cut




###########
#Read Iron#
###########


filename_iron = "C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/iron_alt5200m_qgsii_fluka_r560m_3PMTs_fixed.h5"
print("Reading Iron data: ",filename_iron)

#Read signal at the ground
f_iron = h5py.File(filename_iron, mode = "r")
#Read input data for algorithm
group_iron = f_iron["data"]
#Read input for the algorithm
X_iron = group_iron[()] #X_train: Estimated signal at the ground
#Close file
f_iron.close()


#Read shower parameters stored in the pandas dataframe
InfoDF_iron = pd.read_hdf(filename_iron, key = "info")
E0_iron = InfoDF_iron.iloc[:,1].values

#Compute total signal at the ground for each event (sum of all matrix values)
S_total_iron = X_iron.sum(axis=(1,2))


#Apply the same cut on the signal that was applied for protons: S \in [S_mean-S_std;S_mean+S_std]
index_cut_Signal_iron = (S_total_iron>=S_total_mean-S_total_std)*(S_total_iron<=S_total_mean+S_total_std)



#Keep these events
E0_iron_selected = E0_iron[index_cut_Signal_iron]
X_iron_cut = X_iron[index_cut_Signal_iron]
InfoDF_iron_cut = InfoDF_iron.loc[index_cut_Signal_iron]

#Delete the old data set
del X_iron,InfoDF_iron

#Check number of events within this range 
Nevents_iron = len(X_iron_cut)

#Save new data set if there are events:
outputfilename_iron = output_folder+"iron_alt5200m_qgsii_fluka_r560m_3PMTs_"+sys.argv[1]+"-"+sys.argv[2]+"TeV-GammaERange_N"+str(Nevents_iron)+".h5"


if Nevents_iron>0:
    print("Saving iron data data set: ",outputfilename_iron)
    #Save file with all the data
    with h5py.File(outputfilename_iron, 'w') as hf:
                    hf.create_dataset("data", data=X_iron_cut, compression="gzip", compression_opts=9)
    #Save pandas table with events variables:
    InfoDF_iron_cut.to_hdf(outputfilename_iron, key='info', index=False,mode='a')
    print("Events signal iron: ",Nevents_iron)
    print("Events InfoDF iron: ",len(InfoDF_iron_cut))
    

del X_iron_cut, InfoDF_iron_cut




##################################################################################################################

#################
##Control plots##
#################

#Plot energy range
lim_inf = S_total_mean-S_total_std
lim_sup = S_total_mean+S_total_std
print(lim_inf)
print(lim_sup)

max_limit = np.max([np.max(S_total_gamma_AllEvents),np.max(S_total_proton),np.max(S_total_iron)])

#Compute histograms
n_gamma_all,x_all,_ = plt.hist(S_total_gamma_AllEvents, range=([-1.5,max_limit+2.5]), bins = 400)
n_proton_all,x_all2,_ = plt.hist(S_total_proton, range=([-1.5,max_limit+2.5]), bins = 400)
n_iron_all,x_all3,_ = plt.hist(S_total_iron, range=([-1.5,max_limit+2.5]), bins = 400)


#Keep part of the histogram inside the cut
n_gamma_cut = n_gamma_all[np.intersect1d(np.where(x_all>=lim_inf)[0],np.where(x_all < lim_sup)[0]) ]
x_cut = x_all[np.intersect1d(np.where(x_all>=lim_inf)[0],np.where(x_all < lim_sup)[0]) ]
n_proton_cut = n_proton_all[np.intersect1d(np.where(x_all2>=lim_inf)[0],np.where(x_all2 < lim_sup)[0]) ]
x_cut_2 = x_all2[np.intersect1d(np.where(x_all2>=lim_inf)[0],np.where(x_all2 < lim_sup)[0]) ]
n_iron_cut = n_iron_all[np.intersect1d(np.where(x_all3>=lim_inf)[0],np.where(x_all3 < lim_sup)[0]) ]
x_cut_3 = x_all3[np.intersect1d(np.where(x_all3>=lim_inf)[0],np.where(x_all3 < lim_sup)[0]) ]


#Step histogram
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111)
#Plot histogram in all the range
plt.step(x_all[:-1],n_gamma_all,color="red",where="mid", label = "Fotón")
plt.step(x_all2[:-1],n_proton_all,color="black",where="mid", label="Protón")
plt.step(x_all3[:-1],n_iron_all,color="blue",where="mid", label="Hierro")
#Shaded area below the cut 
plt.fill_between(x_cut,n_gamma_cut, step="mid", alpha=0.2,color="lightcoral")
plt.fill_between(x_cut_2,n_proton_cut, step="mid", alpha=0.28,color="grey")
plt.fill_between(x_cut_3,n_iron_cut, step="mid", alpha=0.15,color="lightskyblue")
#Set legend and axis
plt.autoscale(enable=True)
plt.xlabel('$S_{T}$ [p.e.]', fontsize=18)
plt.ylabel('Frecuencia', fontsize=18)
ax.legend(loc = 'best', edgecolor="black", columnspacing=1,fontsize=20)
#Add box with information:
textstr = '\n'.join((
"Fotón",
'Entradas = %2d' % (Nevents_gamma ),
'Media = %1.1E' % (np.nanmean(S_total_gamma_AllEvents[index_cut_Signal_gamma]), ),
'Desv Est = %1.1E' % (np.nanstd(S_total_gamma_AllEvents[index_cut_Signal_gamma]), )))
textstr2 = '\n'.join((
"Protón",
'Entradas = %2d' % (Nevents_proton, ),
'Media = %1.1E' % (np.nanmean(S_total_proton[index_cut_Signal_proton]), ),
'Desv Est = %1.1E' % (np.nanstd(S_total_proton[index_cut_Signal_proton]), )))
textstr3 = '\n'.join((
"Hierro",
'Entradas = %2d' % (Nevents_iron, ),
'Media = %1.1E' % (np.nanmean(S_total_iron[index_cut_Signal_iron]), ),
'Desv Est = %1.1E' % (np.nanstd(S_total_iron[index_cut_Signal_iron]), )))
props = dict(boxstyle='square', facecolor='white', alpha=0.8)
ax.text(x=0.7, y=0.64, s=textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)
ax.text(x=0.7, y=0.46, s=textstr2, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)
ax.text(x=0.7, y=0.28, s=textstr3, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)
#Set limits and save picture
fig.tight_layout()
ax.set_xlim([1*10**6,2*10**7])
ax.set_ylim([0.1,500])
filename = 'C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/selected_signalevent_'+sys.argv[1]+"-"+sys.argv[2]+'TeV-GammaERange_LinearScale_Cuts.pdf'
plt.savefig(filename)
plt.close()



filename = 'C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/selected_E0event_'+sys.argv[1]+"-"+sys.argv[2]+'TeV-GammaERange_JustSelectedEvents_NormCounts.pdf'
plot_histogram_3vars(data1=E0_gamma_selected/1000,data2=E0_proton_selected/1000,data3=E0_iron_selected/1000,plotname=filename,xlabel_title='$E_{0}$ [TeV]',name1="Fotón",name2="Protón",name3="Hierro",pos="right")


filename = 'C:/Users/andre/OneDrive/Escritorio/Master/TFM/Datos/selected_E0event_'+sys.argv[1]+"-"+sys.argv[2]+'TeV-GammaERange_JustSelectedEvents.pdf'
plot_histogram_3vars(data1=E0_gamma_selected/1000,data2=E0_proton_selected/1000,data3=E0_iron_selected/1000,plotname=filename,xlabel_title='$E_{0}$ [TeV]',name1="Fotón",name2="Protón",name3="Hierro",pos="right",density_option=False)
