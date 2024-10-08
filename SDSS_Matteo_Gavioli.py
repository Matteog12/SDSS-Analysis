import time
start_time=time.time()

import os
import numpy as np
from sys import platform
from copy import deepcopy
from astropy.io import fits
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr

home=os.getcwd()
if platform == 'linux' or platform == 'linux2':
    print("Linux")
    data=home+'/data'

    output=home+'/output'
    out_step_2=output+'/step_2'
    out_step_3=output+'/step_3'
    out_step_4=output+'/step_4'
    out_step_5=output+'/step_5'
    out_step_6=output+'/step_6'

    plots=home+'/plots'
    plot_step_2=plots+'/step_2'
    parent_plot2=plot_step_2+'/parent_plot'
    child_plot2=plot_step_2+'/child_plot'
    plot_step_3=plots+'/step_3'
    trend=plot_step_3+'/trend'
    plot_step_4=plots+'/step_4'
    color_mass_folder=plot_step_4+'/color-mass'
    bpt_folder=plot_step_4+'/BPT'
    sfr_mass_folder=plot_step_4+'/SFR-mass'
    plot_step_5=plots+'/step_5'
    plot_step_6=plots+'/step_6'

elif platform == 'win32':
    print("Windows")
    data=home+'\\data'

    output=home+'\\output'
    out_step_2=output+'\\step_2'
    out_step_3=output+'\\step_3'
    out_step_4=output+'\\step_4'
    out_step_5=output+'\\step_5'
    out_step_6=output+'\\step_6'

    plots=home+'\\plots'
    plot_step_2=plots+'\\step_2'
    parent_plot2=plot_step_2+'\\parent_plot'
    child_plot2=plot_step_2+'\\child_plot'
    plot_step_3=plots+'\\step_3'
    trend=plot_step_3+'\\trend'
    plot_step_4=plots+'\\step_4'
    color_mass_folder=plot_step_4+'\\color-mass'
    bpt_folder=plot_step_4+'\\BPT'
    sfr_mass_folder=plot_step_4+'\\SFR-mass'
    plot_step_5=plots+'\\step_5'
    plot_step_6=plots+'\\step_6'

######################################################################################################
#   Digitare True per sovrascrivere i file e False per non farlo
overwrite=True
#   Digitare True per visualizzare la directory delle cartelle e False per non farlo
display_directory=False
#   ID per estrarre il subsample
myID=27
#   Digitare il nome del file
database='data_SDSS_Info.fit'
#   Numero per cui viene moltiplicata la deviazione standard nel sigma-clipping
Nsig=4
#   Numero delle ripetizioni del ciclo del sigma-clipping
cicles=5
#   Digitare i dpi dei grafici
#   Attenzione: per utilizzare 300 dpi o superiori sono consigliati almeno 8 GB di RAM
DPI=300
#   Numero di volte in cui si utilizzano le mie funzioni per calcolare media e deviazione standard
#   Attenzione: Più il numero è alto maggiore sarà il tempo di esecuzione del programma
times=1
#   Numero di bins degli istogrammi
bins1=50    #Bins parent sample                                     step 2
bins2=30    #Bins subsample                                         step 2
bins3=20    #Bins redshift trend                                    step 3
bins4=15    #Bins variabili sopra e sotto la relazione teorica      step 4
bins5=20    #Bins redshift trend                                    step 5
#   Colore degli istogrammi e dei residui
data_color='#336699'
#   Colore della gaussiana negli istogrammi
gaus_col='g'
#   Colore dei dati dello scatter plot
scatter_color='#336699'
#   Colore della retta di best fit
bf_color='g'
#   Colore delle variabili sopra la relazione teorica dello step 4
up_color='g'
#   Colore delle variabili sotto la relazione teorica dello step 4
down_color='#428af5'
#   Valore minimo del valore assoluto del coefficiente di Pearson
Ncoef=.70
#   Numero di bins in cui si vuole dividere l'array (minimo 1)
Nbins=15
#   Digitare True per visualizzare il coefficiente di Pearson e False per non farlo
print_coefficient=False
#   Inserire la color map desiderata per i grafici 'color-mass', 'BPT', 'SFR-mass'
#   Color map consigliate: 'plasma' , 'inferno' , 'magma'
color_map='plasma'

plt.rcParams["axes.facecolor"]="white"
plt.rcParams["axes.edgecolor"]="black"
plt.rcParams['axes.titlesize']='x-large'
plt.rcParams['axes.labelsize']='large'
#   Dimensione dei titoli dei plot
suptitle_size=20

#   Digitare il divisore da utilizzare
divisore="\n#####################################################################\n"
######################################################################################################

####################
# Funzioni generiche
def folder_creation():
    folder=[data,
            output,out_step_2,out_step_3,out_step_4,out_step_5,out_step_6,
            plots,plot_step_2,parent_plot2,child_plot2,
            plot_step_3,trend,
            plot_step_4,color_mass_folder,bpt_folder,sfr_mass_folder,
            plot_step_5,plot_step_6]
    print("Creazione/controllo delle cartelle in corso")

    for i in folder:
        if os.path.exists(i):
            if display_directory:
                print("la cartella",str(i),"è già esistente.")
        else:
            os.mkdir(i)
            if display_directory:
                print("La cartella",str(i),"è stata creata con successo.")
        os.chdir(home)
    print("Operazione completata con successo.")

def check_file():
    os.chdir(data)
    if os.path.isfile(database):
        print("Il file",str(database),"è già presente nella cartella.\n")
    else:
        print("\nIl file",str(database),"non è presente all'interno della cartella.\nCopia del file in corso.")
        if platform == 'linux' or platform == 'linux2':
            cp='cp ../'+str(database)+' .'
        elif platform == 'win32':
            cp='copy ..\\'+str(database)+' .'
        os.system(cp)
        print("Il file è stato copiato correttamente.\n")
    os.chdir(home)

def file_view():
    os.chdir(data)
    hdulist=fits.open(database)
    cols=hdulist[1].columns
    print("Il file",str(database),"contiene le seguenti colonne:")
    cols.info('name')
    hdulist.close()
    os.chdir(home)

def my_mean(x):
    n=0
    for i in x:
        n+=i
    m=n/len(x)
    return m

def my_std(x):
    m=my_mean(x)
    n=0
    for i in x:
        n+=(i-m)**2
    s=(n/len(x))**(1/2)
    return s

def nan_inf_removal(arr,mode):
    #Pulizia completa su tutti gli array
    if mode==0:
        for e in arr:
            mask=np.isfinite(e)
            n=0
            for i in arr:
                arr[n]=arr[n][mask]
                n+=1

    #Pulizia parziale sulle colonne
    elif mode==1:
        n=0
        for e in arr:
            tmp=e[np.isfinite(e)]
            arr[n]=tmp
            n+=1

    return arr

def sigma_clip(arr,mode):
    #Pulizia completa su tutti gli array
    if mode==0:
        for K in range(cicles):
            for e in arr:
                mask=np.logical_and(e>np.mean(e)-Nsig*np.std(e),e<np.mean(e)+Nsig*np.std(e))
                n=0
                for i in arr:
                    arr[n]=arr[n][mask]
                    n+=1

    #Pulizia parziale sulle colonne
    elif mode==1:
        n=0
        for e in arr:
            for k in range(cicles):
                e=e[np.logical_and(e>np.mean(e)-Nsig*np.std(e),e<np.mean(e)+Nsig*np.std(e))]
                arr[n]=e
            n+=1

    return arr

def cleaning(arr,mode):
    #Uso deepcopy per evitare che mi modifichi anche gli array originali
    new_arr=deepcopy(arr)
    nan_inf_removal(new_arr,mode)
    sigma_clip(new_arr,mode)
    return new_arr

def is_positive(arr,index):
    #index è una lista con gli indici degli array che devono essere positivi
    new_arr=[]
    mask=[True]*len(arr[0])

    for p in index:
        for i in range(len(arr[p])):
            if arr[p][i]<=0:
                mask[i]=False

    for el in arr:
        el=el[mask]
        new_arr.append(el)

    return new_arr

def gaussian(bins,mu,sigma):
    x=np.zeros(len(bins)-1)
    for i in range(len(x)):
        x[i]=(bins[i]+bins[i+1])/2
    y=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
    return x, y

def file_in_folder(filename,directory):
    os.chdir(directory)
    if os.path.isfile(filename):
        return 1
    else:
        return 0
    os.chdir(home)

def check_plot(sample_list,name_list,directory):
    n=0
    sample=[]
    names=[]
    for i in name_list:
        name=i+'.png'
        if file_in_folder(name,directory)==0:
            sample.append(sample_list[n])
            names.append(i)
        n+=1
    return sample, names

def translator(name_arr):
    new_arr=[]
    dict1={'z':'redshift $r$',
        'petroMag_u':'apparent magnitude $m_u$',
        'petroMag_g':'apparent magnitude $m_g$',
        'petroMag_r':'apparent magnitude $m_r$',
        'petroMag_i':'apparent magnitude $m_i$',
        'petroMag_z':'apparent magnitude $m_z$',
        'h_alpha_flux':r'$H\alpha$',
        'h_beta_flux':r'$H\beta$',
        'oiii_5007_flux':'[OIII] $\lambda$5007',
        'nii_6584_flux':'[NII] $\lambda$6584',
        'lgm_tot_p50':'stellar mass',
        'sfr_tot_p50':'SFR',
        'absMagU':'absolute magnitude $M_U$',
        'absMagG':'absolute magnitude $M_G$',
        'absMagR':'absolute magnitude $M_R$',
        'absMagI':'absolute magnitude $M_I$',
        'absMagZ':'absolute magnitude $M_Z$',}

    dict2={'child_z':'subsample redshift $r$',
        'child_petroMag_u':'subsample apparent magnitude $m_u$',
        'child_petroMag_g':'subsample apparent magnitude $m_g$',
        'child_petroMag_r':'subsample apparent magnitude $m_r$',
        'child_petroMag_i':'subsample apparent magnitude $m_i$',
        'child_petroMag_z':'subsample apparent magnitude $m_z$',
        'child_h_alpha_flux':r'subsample $H\alpha$',
        'child_h_beta_flux':r'subsample $H\beta$',
        'child_oiii_5007_flux':'subsample [OIII] $\lambda$5007',
        'child_nii_6584_flux':'subsample [NII] $\lambda$6584',
        'child_lgm_tot_p50':'subsample stellar mass',
        'child_sfr_tot_p50':'subsample SFR',
        'child_absMagU':'subsample absolute magnitude $M_U$',
        'child_absMagG':'subsample absolute magnitude $M_G$',
        'child_absMagR':'subsample absolute magnitude $M_R$',
        'child_absMagI':'subsample absolute magnitude $M_I$',
        'child_absMagZ':'subsample absolute magnitude $M_Z$',}

    dict3={'stellar_mass-sopra':'Stellar Mass log M.(M$_\odot$) sopra la relazione teorica',
            'stellar_mass-sotto':'Stellar Mass log M.(M$_\odot$) sotto la relazione teorica',
            'u-r_colour-sopra':'$u-r$ colour sopra la relazione teorica',
            'u-r_colour-sotto':'$u-r$ colour sotto la relazione teorica',
            'log(NII:Ha)-sopra':r'log ([NII] $\lambda$6584/$H\alpha$) sopra la relazione teorica',
            'log(NII:Ha)-sotto':r'log ([NII] $\lambda$6584/$H\alpha$) sotto la relazione teorica',
            'log(OIII:Hb)-sopra':r'log ([OIII] $\lambda$5007/$H\beta$) sopra la relazione teorica',
            'log(OIII:Hb)-sotto':r'log ([OIII] $\lambda$5007/$H\beta$) sotto la relazione teorica',
            'stellar-mass-sopra':'M$_{star}$, M$_\odot$ sopra la relazione teorica',
            'stellar-mass-sotto':'M$_{star}$, M$_\odot$ sotto la relazione teorica',
            'SFR-sopra':'SFR, M$_\odot$/yr sopra la relazione teorica',
            'SFR-sotto':'SFR, M$_\odot$/yr sotto la relazione teorica'}

    for el in name_arr:
        if el in dict1:
            new_arr.append(dict1[el])
        elif el in dict2:
            new_arr.append(dict2[el])
        elif el in dict3:
            new_arr.append(dict3[el])
        else:
            new_arr.append(el)
    return new_arr

def merge(arr):
    new_arr=[]

    for ar in arr:
        for el in ar:
            new_arr.append(el)

    return new_arr
####################

############################
# Funzioni principali step 2
def fits_creation(data_arr,name_cols_arr,filename,directory):
    os.chdir(directory)
    fileout=filename

    m=[]
    e=[]

    for t in data_arr[:times]:
        m.append(my_mean(t))
        e.append(my_std(t)/(len(t)**(1/2)))

    for i in data_arr[times:]:
        m.append(np.mean(i))
        e.append(np.std(i)/(len(i)**(1/2)))

    col0=fits.Column(name='Nomi',format='A30',array=name_cols_arr)
    col1=fits.Column(name='Media',format='D',array=m)
    col2=fits.Column(name='Errore associato',format='D',array=e)
    cols=fits.ColDefs([col0,col1,col2])
    tbhdu=fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fileout,overwrite=overwrite)
    os.chdir(home)

def output1(parent_data,parent_name,child_data,child_name):
    #per ogni campione creo due file:
    #'media_err_*.fits' dove eseguo la pulizia su ogni singolo array senza togliere le righe corrispondenti degli altri array
    #'media_err_*_v2.fits' dove eseguo la pulizia su tutti gli array, togliendo anche le righe corrispondenti
    names=['media_err_parent.fits','media_err_parent_v2.fits','media_err_child.fits','media_err_child_v2.fits']
    parent_sample=parent_data
    parent_names=parent_name
    child_sample=child_data
    child_names=child_name
    arr=[parent_sample,parent_names,child_sample,child_names]

    if overwrite==False:
        for i in range(len(names)):
            v=file_in_folder(names[i],out_step_2)
            if v==1:
                print("File",str(names[i]),"già esistente.")
            elif v==0:
                print("Creazione del file",str(names[i]),"in corso.")
                if i==0 or i==2:
                    sample=cleaning(arr[i],1)
                    fits_creation(sample,arr[i+1],names[i],out_step_2)
                elif i==1 or i==3:
                    sample=cleaning(arr[i-1],0)
                    fits_creation(sample,arr[i],names[i],out_step_2)
                print("Il file è stato creato con successo.")

    elif overwrite==True:
        print("Creazione dei file di output in corso.")
        for i in range(len(names)):
            if i==0 or i==2:
                sample=cleaning(arr[i],1)
                fits_creation(sample,arr[i+1],names[i],out_step_2)
            elif i==1 or i==3:
                sample=cleaning(arr[i-1],0)
                fits_creation(sample,arr[i],names[i],out_step_2)
        print("I file sono stati creati con successo.")

def histogram(data_arr,name_arr,mode,z_arr):
    names=name_arr
    titles=translator(names)

    if overwrite==False and len(names)==0:
        if mode==0:
            mem="del parent sample"
        elif mode==1:
            mem="del subsample"
        elif mode==2:
            mem="dei trend"
        elif mode==3:
            mem="delle variabili"
        return print("Gli istogrammi "+mem+" sono già stati generati.")

    if mode==0:
        os.chdir(parent_plot2)
        binning=bins1
        data=cleaning(data_arr,1)
        mem="del parent sample"
    elif mode==1:
        os.chdir(child_plot2)
        binning=bins2
        data=cleaning(data_arr,1)
        mem="del subsample"
    elif mode==2:
        data=data_arr
        os.chdir(trend)
        titles=[]
        z_names=translator(z_arr[0])
        z_val=z_arr[1]
        for i in z_names:
            el1=i+': '+str(round(z_val[0],4))+'<=$z$<='+str(round(z_val[1],4))
            el2=i+': '+str(round(z_val[1],4))+'<$z$<='+str(round(z_val[2],4))
            el3=i+': '+str(round(z_val[2],4))+'<$z$<='+str(round(z_val[3],4))
            titles.append(el1)
            titles.append(el2)
            titles.append(el3)
        binning=bins3
        mem="dei trend"
    elif mode==3:
        data=data_arr
        binning=bins4
        mem="delle variabili"

    print("Creazione degli istogrammi "+mem+" in corso.\nL'operazione potrebbe richiedere del tempo, si prega di attendere.")

    gs=gridspec.GridSpec(1,3)

    for i in range(len(data)):
        fig=plt.figure(i+1,figsize=(17,10))
        var=data[i]

        ax1=plt.subplot(gs[0])
        counts,bins,ign=ax1.hist(var,binning,histtype='bar',density=True,align='mid',color=data_color,label='grafico\nnormalizzato',zorder=2,alpha=.8)
        xm,mod=gaussian(bins,np.mean(var),np.std(var))
        ax1.plot(xm,mod,c=gaus_col,ls='--',lw=2,label='gaussiana')
        ax1.axvline(np.mean(var),c='.3',ls='-.',label='media')
        a=np.mean(var)-np.std(var)
        b=np.mean(var)+np.std(var)
        ax1.axvspan(a,b,color='.3',zorder=1,alpha=.25)
        ax1.legend(loc='best',fontsize='medium')
        ax1.set_title('Media')
        ax1.set_xlabel('Valore della variabile')
        ax1.set_ylabel('Frequenza normalizzata')

        ax2=plt.subplot(gs[1])
        ax2.hist(var,binning,histtype='bar',density=True,align='mid',color=data_color,label='grafico\nnormalizzato',zorder=2,alpha=.8)
        ax2.axvline(np.median(var),c='.3',ls='-.',label='mediana')
        a=np.median(var)-np.std(var)
        b=np.median(var)+np.std(var)
        ax2.axvspan(a,b,color='.3',zorder=1,alpha=.25)
        ax2.legend(loc='best',fontsize='medium')
        ax2.set_title('Mediana')
        ax2.set_xlabel('Valore della variabile')
        ax2.set_ylabel('Frequenza normalizzata')

        ax3=plt.subplot(gs[2])
        resid=(counts-mod)
        ax3.scatter(xm,resid,color=data_color,edgecolor='k',label='residui',zorder=3)
        ax3.axhline(np.mean(resid),c='.3',ls='--',label='media',zorder=2)
        a=np.mean(resid)-np.std(resid)
        b=np.mean(resid)+np.std(resid)
        ax3.axhspan(a,b,color='.3',alpha=.25,zorder=1)
        ax3.legend(loc='best',fontsize='medium')
        ax3.set_title('Residui')
        ax3.set_xlabel('Valore della variabile')
        ax3.set_ylabel('Scarto')
        ax3.yaxis.tick_right()

        fig.suptitle(str(titles[i]),fontsize=suptitle_size,y=.955)
        plt.savefig(names[i]+'.png',dpi=DPI)
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(names)),"istogrammi.")
    print("Processo completato con successo.")

    os.chdir(home)
############################

############################
# Funzioni principali step 3
def scatter_plot(data_arr,name_arr):
    if overwrite==False and len(name_arr)==1:
        return print("I grafici di correlazione sono già stati generati.\n")

    data=cleaning(data_arr,0)
    titles=translator(name_arr)
    os.chdir(plot_step_3)

    redshift=data[0]
    print("Creazione dei grafici di correlazione in corso.\nL'operazione potrebbe richiedere del tempo, si prega di attendere.")
    for i in range(len(name_arr)-1):
        fig=plt.figure(i+1,figsize=(15,10))

        l=data[i+1]

        res=np.polyfit(redshift,l,1)
        y_bf=np.polyval(res,redshift)

        plt.scatter(redshift,l,color=scatter_color,edgecolors='k',label='data')
        plt.plot(redshift,y_bf,color=bf_color,lw=2,label='best-fit',alpha=.8)
        plt.legend(loc='best',fontsize='medium')

        fig.suptitle(str(titles[i+1]+" - "+str(titles[0])),fontsize=suptitle_size,y=.955)
        plt.xlabel(str(titles[0]),fontsize='x-large')
        plt.ylabel(str(titles[i+1]),fontsize='x-large')
        plt.savefig(name_arr[i+1]+'.png',dpi=DPI)
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(name_arr)-1),"grafici.")
    print("Processo completato\n")

    os.chdir(home)

def correlation(data_arr,name_arr):
    sample=data_arr

    data=cleaning(sample,0)
    temp=[]
    names=[]
    z_range1=[]

    for i in range(len(sample)-1):
        l=data[i+1]
        z_arr=data[0]
        coef,p_value=pearsonr(z_arr,l)
        if print_coefficient:
            print("Coefficiente di correlazione per",str(name_arr[i+1]),"r="+str(coef))
        if coef<-Ncoef or coef>Ncoef:
            temp.append(l)
            names.append(name_arr[i+1]+'_trend1')
            names.append(name_arr[i+1]+'_trend2')
            names.append(name_arr[i+1]+'_trend3')
            z_range1.append(name_arr[i+1])

    if len(temp)==0:
        return print("Nessuna correlazione è stata rilevata.")

    if print_coefficient:
        print()

    tmp=(max(z_arr)-min(z_arr))/3
    l1=tmp+min(z_arr)
    l2=2*tmp+min(z_arr)
    plotting_list=[]
    z_range2=[min(z_arr),l1,l2,max(z_arr)]
    z_range=[z_range1,z_range2]
    for e in temp:
        range1=[]
        range2=[]
        range3=[]
        n=0
        for i in z_arr:
            if i<=l1:
                range1.append(e[n])
            elif l1<i<=l2:
                range2.append(e[n])
            else:
                range3.append(e[n])
            n+=1
        plotting_list.append(range1)
        plotting_list.append(range2)
        plotting_list.append(range3)

    return plotting_list,names,z_range

def legend_val(z_arr,directory):
    #Creo un file fits contenente il range del redshift secondo cui gli array sono suddivisi nei vari bins
    os.chdir(directory)
    filename='range_legend.fits'

    names=['trend1[min<=z<=max]','trend2[min<z<=max]','trend3[min<z<=max]']
    minimum=[z_arr[0],z_arr[1],z_arr[2]]
    maximum=[z_arr[1],z_arr[2],z_arr[3]]

    col0=fits.Column(name='Trend',format='A25',array=names)
    col1=fits.Column(name='Valore minimo di z',format='D',array=minimum)
    col2=fits.Column(name='Valore massimo di z',format='D',array=maximum)
    cols=fits.ColDefs([col0,col1,col2])
    tbhdu=fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(filename,overwrite=overwrite)
    os.chdir(home)

def trend_subplot(data_arr,name_arr):
    try:
        plotting_list,names,z_range=correlation(data_arr,name_arr)
        filename='media_err_trend.fits'
        directory=out_step_3

        if overwrite==False:
            v1=file_in_folder(filename,directory)
            v2=file_in_folder('range_legend.fits',directory)
            if v1==1:
                print("File",filename,"già esistente.")
                if v2==0:
                    legend_val(z_range[1],directory)
            elif v1==0:
                print("Creazione del file",filename,"in corso.")
                fits_creation(plotting_list,names,filename,directory)
                if v2==0:
                    legend_val(z_range[1],directory)
                print("Il file di output è stato creato con successo.")
            plotting_list,names=check_plot(plotting_list,names,trend)
            print()
            histogram(plotting_list,names,2,z_range)

        elif overwrite==True:
            print("Creazione dei file di output in corso.")
            fits_creation(plotting_list,names,filename,directory)
            legend_val(z_range[1],directory)
            print("I file sono stati creati con successo.\n")
            histogram(plotting_list,names,2,z_range)
    except:
        print("I grafici e gli output non sono stati generati poiché nessuna correlazione è stata rilevata.")
############################

############################
# Funzioni principali step 4
def arr_sep(data_arr,name_arr):
    hist_data=[]
    scatter_data=[]

    for i in range(len(name_arr)):
        data=cleaning(data_arr[i],0)

        if name_arr[i]=='color-mass':
            x_var=data[2]
            y_var=data[0]-data[1]
            #(u-r)=-.495+.25*mass
            x_tr=np.arange(min(x_var),max(x_var),.1)
            y_tr=-.495+.25*x_tr
            mask=y_var>-.495+.25*x_var

        elif name_arr[i]=='BPT':
            pos=[0,1,2,3]
            data=is_positive(data,pos)

            x_var=np.log10(data[2]/data[0])
            y_var=np.log10(data[3]/data[1])
            #log([OIII]/Hb)>.61/(log([NII]/Ha)-.05)+1.3
            x_tmp=np.arange(min(x_var),max(x_var),.1)
            y_tmp=.61/(x_tmp-.05)+1.3
            M=max(x_var)
            for t in range(len(y_tmp)):
                if y_tmp[t]<=min(y_var):
                    M=x_tmp[t]
                    break
            x_tr=np.arange(min(x_var),M,.1)
            y_tr=.61/(x_tr-.05)+1.3
            mask=np.logical_or((y_var>.61/(x_var-.05)+1.3),(x_var>=M))

        elif name_arr[i]=='SFR-mass':
            x_var=data[1]
            y_var=data[0]
            #SFR=-8.64+.76*mass
            x_tr=np.arange(min(x_var),max(x_var),.1)
            y_tr=-8.64+.76*x_tr
            mask=y_var>-8.64+.76*x_var

        redshift=data[len(data)-1]
        temp=[x_var,y_var,x_tr,y_tr,redshift]
        scatter_data.append(temp)

        x_above=x_var[mask]
        x_below=x_var[~mask]
        y_above=y_var[mask]
        y_below=x_var[~mask]

        tmp=[x_above,x_below,y_above,y_below]
        hist_data.append(tmp)

    return scatter_data,hist_data

def diagram(data_arr,name_arr):
    if overwrite==False and len(name_arr)==0:
        return print("I diagrammi sono già stati generati.")

    print("Creazione dei diagrammi in corso.\nL'operazione potrebbe richiedere del tempo, si prega di attendere.")

    for i in range(len(name_arr)):

        if name_arr[i]=='color-mass-diagramma':
            os.chdir(color_mass_folder)
            names=['Stellar Mass log M.(M$_\odot$)','$u-r$ colour','color-mass diagram']

        elif name_arr[i]=='BPT-diagramma':
            os.chdir(bpt_folder)
            names=[r'log ([NII] $\lambda$6584/$H\alpha$)',r'log ([OIII] $\lambda$5007/$H\beta$)','BPT diagram']

        elif name_arr[i]=='SFR-mass-diagramma':
            os.chdir(sfr_mass_folder)
            names=['M$_{star}$, M$_\odot$','SFR, M$_\odot$/yr','SFR-mass diagram']

        x_var=data_arr[i][0]
        y_var=data_arr[i][1]
        x_tr=data_arr[i][2]
        y_tr=data_arr[i][3]
        redshift=data_arr[i][4]

        fig=plt.figure(i,figsize=(15,10))

        sc=plt.scatter(x_var,y_var,c=redshift,cmap=color_map)
        plt.xlabel(names[0],fontsize='x-large')
        plt.ylabel(names[1],fontsize='x-large')
        cb=plt.colorbar(sc)
        cb.set_ticks(np.arange(0,.1,.01))
        cb.set_label(r'$redshift$')
        plt.plot(x_tr,y_tr,ls='--',label='relazione\nteorica')
        fig.suptitle(names[2],fontsize=suptitle_size,y=.955)
        plt.legend()
        plt.savefig(name_arr[i]+'.png',dpi=DPI)
        plt.close()

        print("Creati",str(i+1)+"/"+str(len(name_arr)),"grafici.")
    print("Processo completato con successo.")

    os.chdir(home)

def general(data_arr,name_arr):
    if overwrite==False and len(name_arr)==0:
        return print("Gli istogrammi delle variabili 'x' e 'y' sono già stati generati.")

    print("Creazione degli istogrammi delle variabili 'x' e 'y' in corso.\nL'operazione potrebbe richiedere del tempo, si prega di attendere.")
    for i in range(len(name_arr)):
        if name_arr[i]=='color-mass-generale':
            os.chdir(color_mass_folder)
            x_var='Stellar Mass log M.(M$_\odot$)'
            y_var='$u-r$ colour'
        elif name_arr[i]=='BPT-generale':
            os.chdir(bpt_folder)
            x_var=r'log ([NII] $\lambda$6584/$H\alpha$)'
            y_var=r'log ([OIII] $\lambda$5007/$H\beta$)'
        elif name_arr[i]=='SFR-mass-generale':
            os.chdir(sfr_mass_folder)
            x_var='M$_{star}$, M$_\odot$'
            y_var='SFR, M$_\odot$/yr'

        fig=plt.figure(i,figsize=(20,10))
        gs=plt.GridSpec(1,4,wspace=.55)
        gs0=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,:2],wspace=0)
        gs1=gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,2:],wspace=0)

        ax1=plt.subplot(gs0[0])
        ax2=plt.subplot(gs0[1])
        ax3=plt.subplot(gs0[2])
        ax4=plt.subplot(gs1[0])
        ax5=plt.subplot(gs1[1])
        ax6=plt.subplot(gs1[2])
        grid=[[ax1,ax2,ax3],[ax4,ax5,ax6]]

        for n in range(len(data_arr[i])):
            var=data_arr[i][n]
            if n<=1:
                g=0
            else:
                g=1

            if n%2==0:
                order=[2,4,6,8,10]
                location='sopra'
                colour=up_color
            else:
                order=[1,3,5,7,9]
                colour=down_color
                location='sotto'

            counts,bins,ign=grid[g][0].hist(var,bins=bins4,histtype='bar',density=True,align='mid',color=colour,zorder=order[1],alpha=.2)
            xm,mod=gaussian(bins,np.mean(var),np.std(var))
            grid[g][0].hist(var,bins=bins4,color=colour,histtype='step',density=True,align='mid',zorder=order[2])
            grid[g][0].axvline(np.mean(var),c=colour,ls='-.',lw=1,label='media\nvariabile\n'+str(location),zorder=order[4])
            grid[g][0].legend(loc='best',fontsize='small')
            grid[g][0].set_title('Media')
            grid[g][0].set_xlabel('Valore della variabile')
            grid[g][0].set_ylabel('Frequenza normalizzata')

            grid[g][1].hist(var,bins=bins4,density=True,histtype='bar',align='mid',color=colour,zorder=order[1],alpha=.2)
            grid[g][1].hist(var,bins=bins4,color=colour,histtype='step',density=True,align='mid',zorder=order[2])
            grid[g][1].axvline(np.median(var),c=colour,ls='-.',lw=1,label='mediana\nvariabile\n'+str(location),zorder=order[3])
            grid[g][1].legend(loc='best',fontsize='small')
            grid[g][1].set_title('Mediana')
            grid[g][1].set_xlabel('Valore della variabile')
            grid[g][1].set_yticklabels([])

            resid=(counts-mod)
            grid[g][2].scatter(xm,resid,c=colour,edgecolor='k',label='residui\n'+str(location),zorder=order[1],alpha=.65)
            grid[g][2].axhline(np.mean(resid),c=colour,ls='--',label='media\nresidui\n'+str(location),zorder=order[2])
            grid[g][2].legend(loc='best',fontsize='small')
            grid[g][2].set_title('Residui')
            grid[g][2].set_xlabel('Valore della variabile')
            grid[g][2].set_ylabel('Scarto')
            grid[g][2].yaxis.set_label_position('right')
            grid[g][2].yaxis.tick_right()

        fig.text(x=0.15,y=0.955,s=x_var,horizontalalignment='left',verticalalignment='top',fontsize=suptitle_size)
        fig.text(x=0.58,y=0.955,s=y_var,horizontalalignment='left',verticalalignment='top',fontsize=suptitle_size)
        plt.savefig(name_arr[i]+'.png')
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(name_arr)),"grafici.")
    print("Processo completato con successo.")

    os.chdir(home)

def var_subplot(data,diagrams):
    hist_data=data
    names=[['stellar_mass-sopra','stellar_mass-sotto','u-r_colour-sopra','u-r_colour-sotto'],
            ['log(NII_on_Ha)-sopra','log(NII_on_Ha)-sotto','log(OIII_on_Hb)-sopra','log(OIII_on_Hb)-sotto'],
            ['stellar-mass-sopra','stellar-mass-sotto','SFR-sopra','SFR-sotto']]
    folder=[color_mass_folder,bpt_folder,sfr_mass_folder]

    for i in range(len(names)):
        print(diagrams[i]+":")
        if overwrite==False:
            data_i,name_i=check_plot(hist_data[i],names[i],folder[i])
        else:
            data_i,name_i=hist_data[i],names[i]

        os.chdir(folder[i])
        histogram(data_i,name_i,3,None)
        print()

    data_arr=[]
    names_arr=[]
    for f1 in hist_data:
        for arr in f1:
            data_arr.append(arr)
    for f2 in names:
        for n in f2:
            names_arr.append(n)

    filename='media_err_diagrammi.fits'
    v=file_in_folder(filename,out_step_4)
    if overwrite==True or v==0:
        print("Creazione del file",filename,"in corso.")
        fits_creation(data_arr,names_arr,filename,out_step_4)
        print("Il file è stato creato con successo.")
    else:
        print("File",filename,"già esistente.")
############################

############################
# Funzioni principali step 5
def z_div(z_arr,nbins):
    lenght=(max(z_arr)-min(z_arr))/nbins
    z_range=[]

    for t in range(nbins):
        z_tmp=[]
        for el in z_arr:
            if t==0:
                if el<=min(z_arr)+lenght:
                    z_tmp.append(el)
            elif t==nbins-1:
                if min(z_arr)+(t)*lenght<el<=max(z_arr):
                    z_tmp.append(el)
            else:
                if min(z_arr)+(t)*lenght<el<=min(z_arr)+(t+1)*lenght:
                    z_tmp.append(el)
        z_range.append(z_tmp)

    return z_range

def bins_sep(data_arr,z_arr,z_range):

    div_arr=[]
    for ar in z_range:
        tmp=[]
        for i in range(len(z_arr)):
            if z_arr[i] in ar:
                tmp.append(data_arr[i])
        div_arr.append(tmp)

    return div_arr

def coordinates(z_range,bins_arr):
    x_arr=[]
    y_arr=[]
    for el in z_range:
        x_arr.append(np.mean(el))

    for t in bins_arr:
        y_arr.append(np.mean(t))

    return x_arr,y_arr

def plotta(data_arr,name_arr):
    if overwrite==False and len(name_arr)==1:
        return print()

    os.chdir(plot_step_5)
    data=cleaning(data_arr,0)
    redshift=data[0]
    hist_arr,ign,z_range=correlation(data_arr,name_arr)
    names=z_range[0]

    data=[]
    counter=0
    for k in range(len(names)):
        tmp=hist_arr[k]+hist_arr[k+1]+hist_arr[k+2]
        data.append(tmp)

    bins_arr=[]
    for el in data:
        temp=bins_sep(data,redshift,Nbins)
        x_arr,y_arr,err_down,err_up=coordinates(redshift,temp)
        bins_arr.append([x_arr,y_arr,err_down,err_up])

    names.append(name_arr[0])
    titles=translator(names)
    lab1=str(round(z_range[1][0],4))+'<=$z$<='+str(round(z_range[1][1],4))
    lab2=str(round(z_range[1][1],4))+'<$z$<='+str(round(z_range[1][2],4))
    lab3=str(round(z_range[1][2],4))+'<$z$<='+str(round(z_range[1][3],4))
    lab=[lab1,lab2,lab3]
    colour=['#ff8000','g','#00bfff']

    n=0
    for i in range(len(names)-1):
        l=data[i]

        res=np.polyfit(redshift,l,1)
        y_bf=np.polyval(res,redshift)

        fig=plt.figure(i,figsize=(17,10))
        gs=plt.GridSpec(1,2,wspace=0)
        ax1=plt.subplot(gs[0])
        ax1.scatter(redshift,l,edgecolor='k',label='data')
        ax1.plot(redshift,y_bf,c='r',lw=2,label='best fit',alpha=.8)
        ax1.errorbar(bins_arr[i][0],bins_arr[i][1])
        ax1.legend(loc='best',fontsize='medium')
        ax1.set_xlabel(titles[0])
        ax1.set_ylabel(titles[i+1])

        ax2=plt.subplot(gs[1])
        for t in range(3):
            ax2.hist(hist_arr[n+t],bins=bins5,histtype='bar',density=True,align='mid',color=colour[t],orientation='horizontal',alpha=.2)
        for t in range(3):
            ax2.hist(hist_arr[n+t],bins=bins5,histtype='step',density=True,align='mid',color=colour[t],orientation='horizontal',label=lab[t])
        ax2.legend(loc='upper right',fontsize='medium')
        ax2.set_xlabel('Frequenza normalizzata')
        ax2.yaxis.tick_right()

        fig.suptitle(str(titles[i]+" - "+str(titles[len(names)-1])),fontsize=suptitle_size,y=.955)
        plt.savefig(names[i]+'.png',dpi=DPI)
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(names)-1),"grafici.")
        n+=3
    print("Processo completato")
    os.chdir(home)

def plots3(data_arr,name_arr):
    if overwrite==False and len(name_arr)==1:
        return print("I grafici sono già stati generati.")

    os.chdir(plot_step_5)
    data=cleaning(data_arr,0)
    redshift=data[0]
    titles=translator(name_arr)
    hist_arr,ign,z_range=correlation(data,name_arr)

    lab1=str(round(z_range[1][0],4))+'<=$z$<='+str(round(z_range[1][1],4))
    lab2=str(round(z_range[1][1],4))+'<$z$<='+str(round(z_range[1][2],4))
    lab3=str(round(z_range[1][2],4))+'<$z$<='+str(round(z_range[1][3],4))
    lab=[lab1,lab2,lab3]
    colour=['#ff8000','g','#00bfff']
    order=[4,1,5,2,6,3]

    n=0
    for i in range(len(name_arr)-1):
        l=data[i+1]

        res=np.polyfit(redshift,l,1)
        y_bf=np.polyval(res,redshift)

        fig=plt.figure(i,figsize=(17,10))
        gs=plt.GridSpec(1,2,wspace=0)
        ax1=plt.subplot(gs[0])
        ax1.scatter(redshift,l,edgecolor='k',label='data')
        ax1.plot(redshift,y_bf,c='r',lw=2,label='best fit',alpha=.8)
        ax1.legend(loc='best',fontsize='medium')
        ax1.set_xlabel(titles[0])
        ax1.set_ylabel(titles[i+1])

        ax2=plt.subplot(gs[1])
        for t in range(3):
            ax2.hist(hist_arr[n+t],bins=bins5,histtype='bar',density=True,align='mid',color=colour[t],orientation='horizontal',alpha=.2)
        for t in range(3):
            ax2.hist(hist_arr[n+t],bins=bins5,histtype='step',density=True,align='mid',color=colour[t],orientation='horizontal',label=lab[t])
        ax2.legend(loc='upper right',fontsize='medium')
        ax2.set_xlabel('Frequenza normalizzata')
        ax2.yaxis.tick_right()

        fig.suptitle(str(titles[i+1]+" - "+str(titles[0])),fontsize=suptitle_size,y=.955)
        plt.savefig(name_arr[i+1]+'.png',dpi=DPI)
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(name_arr)-1),"grafici.")
        n+=3
    print("Processo completato")
    os.chdir(home)

############################

############################
# Funzioni principali step 6
def general_corr(data_arr,name_arr):
    names=name_arr
    data=[]
    name_arr=[]
    vals=[]
    for i in range(len(data_arr)-1):
        new_arr=deepcopy(data_arr)
        new_name=deepcopy(names)
        x_var=data_arr[i]
        tmp=new_arr[:i]+new_arr[i+1:]
        new_name=new_name[:i]+new_name[i+1:]
        for k in range(len(tmp)):
            y_var=tmp[k]
            coef,p_value=pearsonr(x_var,y_var)
            if coef<=-Ncoef or coef>=Ncoef:
                count=0
                for n in name_arr:
                    if names[i] in n and new_name[k] in n:
                        count=1
                        break
                if count==0:
                    name_arr.append([names[i],new_name[k]])
                    data.append([x_var,y_var])
                    vals.append([coef,p_value])
                    if print_coefficient:
                        print(names[i]+" - "+new_name[k]+": "+str(coef))

    return name_arr,vals

def corr_type(vals,name_arr):
    type_arr=[]
    per_arr=[]

    for i in range(len(vals)):
        coef=vals[i][0]
        p1=round((coef/1)*100,2)
        p2=round((coef/-1)*100,2)
        percentage=max([p1,p2])
        if coef>0:
            c_type="correlazione"
        else:
            c_type="anticorrelazione"
        type_arr.append(c_type)
        per_arr.append(percentage)

        if print_coefficient:
            print("Le varibaili "+str(name_arr[i][0])+" - "+str(name_arr[i][1])+" sono in un rapporto di "+str(c_type)+" pari al "+str(percentage)+"%"
                "\ncon un coefficiente di Pearson pari a: "+str(round(coef,2))+".")

    return type_arr,per_arr

def corr_fits(name_arr,vals,type_arr,per_arr,filename):
    os.chdir(out_step_6)
    fileout=filename

    r_arr=[]
    p_arr=[]
    val1=[]
    val2=[]

    for i in range(len(name_arr)):
        val1.append(name_arr[i][0])
        val2.append(name_arr[i][1])
        r_arr.append(vals[i][0])
        p_arr.append(vals[i][1])

    col0=fits.Column(name='Variabile 1',format='A30',array=val1)
    col1=fits.Column(name='Variabile 2',format='A30',array=val2)
    col2=fits.Column(name='Coefficiente di Pearson',format='D',array=r_arr)
    col3=fits.Column(name='p-value',format='D',array=p_arr)
    col4=fits.Column(name='Tipo di correlazione',format='A16',array=type_arr)
    col5=fits.Column(name='Percentuale',format='D',array=per_arr)
    cols=fits.ColDefs([col0,col1,col2,col3,col4,col5])
    tbhdu=fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fileout,overwrite=overwrite)
    os.chdir(home)

def corr_plots(data_arr,name_arr):
    if overwrite==False and len(name_arr)==0:
        return print("I grafici di correlazione sono già stati generati.\n")

    os.chdir(plot_step_6)
    print(data_arr)

    print("Creazione dei grafici di correlazione in corso.\nL'operazione potrebbe richiedere del tempo, si prega di attendere.")
    for i in range(len(name_arr)):
        fig=plt.figure(i+1,figsize=(15,10))

        val1=data[i][0]
        val2=data[i][1]
        #print(name_arr[i])
        val_names=translator(name_arr[i])

        res=np.polyfit(val1,val2,1)
        y_bf=np.polyval(res,val1)

        plt.scatter(val1,val2,edgecolors='k',label='data')
        plt.plot(val1,y_bf,color='g',lw=2,label='best-fit',alpha=.8)
        plt.legend(loc='best',fontsize='medium')

        fig.suptitle(str(val_names[1])+" - "+str(val_names[0]),fontsize=suptitle_size,y=.955)
        plt.xlabel(str(val_names[0]),fontsize='x-large')
        plt.ylabel(str(val_names[1]),fontsize='x-large')
        plt.savefig(name_arr[i][0]+name_arr[i][1]+'.png',dpi=DPI)
        plt.close()
        print("Creati",str(i+1)+"/"+str(len(name_arr)),"grafici.")
    print("Processo completato\n")

    os.chdir(home)
############################

print(divisore+
        "\nL'opzione di 'overwrite' è stata impostata su: "+str(overwrite)+"\nID selezionato: "+str(myID)+"\n"
        +divisore+
        "Step I: Lettura del catalogo e selezione del sottocampione\n")

folder_creation()
check_file()
file_view()

os.chdir(data)
hdulist=fits.open(database)
values=hdulist[1].data

if True:
    #####parent sample#####
    specobjid=values['specobjid']
    plate=values['plate']
    mjd=values['mjd']
    fiberid=values['fiberid']
    ra=values['ra']
    dec=values['dec']
    z=values['z']
    petroMag_u=values['petroMag_u']
    petroMagErr_u=values['petroMagErr_u']
    petroMag_g=values['petroMag_g']
    petroMagErr_g=values['petroMagErr_g']
    petroMag_r=values['petroMag_r']
    petroMagErr_r=values['petroMagErr_r']
    petroMag_i=values['petroMag_i']
    petroMagErr_i=values['petroMagErr_i']
    petroMag_z=values['petroMag_z']
    petroMagErr_z=values['petroMagErr_z']
    h_alpha_flux=values['h_alpha_flux']
    h_alpha_flux_err=values['h_alpha_flux_err']
    h_beta_flux=values['h_beta_flux']
    h_beta_flux_err=values['h_beta_flux_err']
    oiii_5007_flux=values['oiii_5007_flux']
    oiii_5007_flux_err=values['oiii_5007_flux_err']
    nii_6584_flux=values['nii_6584_flux']
    nii_6584_flux_err=values['nii_6584_flux_err']
    lgm_tot_p50=values['lgm_tot_p50']
    lgm_tot_p16=values['lgm_tot_p16']
    lgm_tot_p84=values['lgm_tot_p84']
    sfr_tot_p50=values['sfr_tot_p50']
    sfr_tot_p16=values['sfr_tot_p16']
    sfr_tot_p84=values['sfr_tot_p84']
    absMagU=values['absMagU']
    absMagG=values['absMagG']
    absMagR=values['absMagR']
    absMagI=values['absMagI']
    absMagZ=values['absMagZ']
    ID=values['ID']

    #####subsample#####
    child_specobjid=specobjid[ID==myID]
    child_plate=plate[ID==myID]
    child_mjd=mjd[ID==myID]
    child_fiberid=fiberid[ID==myID]
    child_ra=ra[ID==myID]
    child_dec=dec[ID==myID]
    child_z=z[ID==myID]
    child_petroMag_u=petroMag_u[ID==myID]
    child_petroMagErr_u=petroMagErr_u[ID==myID]
    child_petroMag_g=petroMag_g[ID==myID]
    child_petroMagErr_g=petroMagErr_g[ID==myID]
    child_petroMag_r=petroMag_r[ID==myID]
    child_petroMagErr_r=petroMagErr_r[ID==myID]
    child_petroMag_i=petroMag_i[ID==myID]
    child_petroMagErr_i=petroMagErr_i[ID==myID]
    child_petroMag_z=petroMag_z[ID==myID]
    child_petroMagErr_z=petroMagErr_z[ID==myID]
    child_h_alpha_flux=h_alpha_flux[ID==myID]
    child_h_alpha_flux_err=h_alpha_flux_err[ID==myID]
    child_h_beta_flux=h_beta_flux[ID==myID]
    child_h_beta_flux_err=h_beta_flux_err[ID==myID]
    child_oiii_5007_flux=oiii_5007_flux[ID==myID]
    child_oiii_5007_flux_err=oiii_5007_flux_err[ID==myID]
    child_nii_6584_flux=nii_6584_flux[ID==myID]
    child_nii_6584_flux_err=nii_6584_flux_err[ID==myID]
    child_lgm_tot_p50=lgm_tot_p50[ID==myID]
    child_lgm_tot_p16=lgm_tot_p16[ID==myID]
    child_lgm_tot_p84=lgm_tot_p84[ID==myID]
    child_sfr_tot_p50=sfr_tot_p50[ID==myID]
    child_sfr_tot_p16=sfr_tot_p16[ID==myID]
    child_sfr_tot_p84=sfr_tot_p84[ID==myID]
    child_absMagU=absMagU[ID==myID]
    child_absMagG=absMagG[ID==myID]
    child_absMagR=absMagR[ID==myID]
    child_absMagI=absMagI[ID==myID]
    child_absMagZ=absMagZ[ID==myID]

hdulist.close()
os.chdir(home)

print(divisore[:-1])

def step2():
    parent_sample=[z,
        petroMag_u,petroMag_g,petroMag_r,petroMag_i,petroMag_z,
        h_alpha_flux,h_beta_flux,oiii_5007_flux,nii_6584_flux,
        lgm_tot_p50,sfr_tot_p50,
        absMagU,absMagG,absMagR,absMagI,absMagZ]

    parent_names=['z',
        'petroMag_u','petroMag_g','petroMag_r','petroMag_i','petroMag_z',
        'h_alpha_flux','h_beta_flux','oiii_5007_flux','nii_6584_flux',
        'lgm_tot_p50','sfr_tot_p50',
        'absMagU','absMagG','absMagR','absMagI','absMagZ']

    child_sample=[child_z,
        child_petroMag_u,child_petroMag_g,child_petroMag_r,child_petroMag_i,child_petroMag_z,
        child_h_alpha_flux,child_h_beta_flux,child_oiii_5007_flux,child_nii_6584_flux,
        child_lgm_tot_p50,child_sfr_tot_p50,
        child_absMagU,child_absMagG,child_absMagR,child_absMagI,child_absMagZ]

    child_names=['child_z',
        'child_petroMag_u','child_petroMag_g','child_petroMag_r','child_petroMag_i','child_petroMag_z',
        'child_h_alpha_flux','child_h_beta_flux','child_oiii_5007_flux','child_nii_6584_flux',
        'child_lgm_tot_p50','child_sfr_tot_p50',
        'child_absMagU','child_absMagG','child_absMagR','child_absMagI','child_absMagZ']

    output1(parent_sample,parent_names,child_sample,child_names)

    if overwrite==False:
        parent_sample,parent_names=check_plot(parent_sample,parent_names,parent_plot2)
        child_sample,child_names=check_plot(child_sample,child_names,child_plot2)

    print()
    histogram(parent_sample,parent_names,0,None)
    print()
    histogram(child_sample,child_names,1,None)

def step3():
    child_sample=[child_z,child_petroMag_u,child_h_alpha_flux,child_lgm_tot_p50,child_sfr_tot_p50,child_absMagU]
    child_names=['child_z','child_petroMag_u','child_h_alpha_flux','child_lgm_tot_p50','child_sfr_tot_p50','child_absMagU']

    if overwrite==False:
        child_sample1,child_names1=check_plot(child_sample,child_names,plot_step_3)
        scatter_plot(child_sample1,child_names1)
        trend_subplot(child_sample,child_names)
        return

    scatter_plot(child_sample,child_names)
    trend_subplot(child_sample,child_names)

def step4():
    color_mass=[child_petroMag_u,child_petroMag_r,child_lgm_tot_p50,child_z]
    bpt=[child_h_alpha_flux,child_h_beta_flux,child_nii_6584_flux,child_oiii_5007_flux,child_z]
    sfr=[child_sfr_tot_p50,child_lgm_tot_p50,child_z]

    data=[color_mass,bpt,sfr]
    diagrams=['color-mass','BPT','SFR-mass']
    diagram_names=['color-mass-diagramma','BPT-diagramma','SFR-mass-diagramma']
    general_names=['color-mass-generale','BPT-generale','SFR-mass-generale']

    scatter_data,hist_data=arr_sep(data,diagrams)

    if overwrite==False:
        scatter_data1,diagram_names1=check_plot(scatter_data,diagram_names,color_mass_folder)
        hist_data1,general_names1=check_plot(hist_data,general_names,color_mass_folder)
        scatter_data1,diagram_names1=check_plot(scatter_data1,diagram_names1,bpt_folder)
        hist_data1,general_names1=check_plot(hist_data1,general_names1,bpt_folder)
        scatter_data1,diagram_names1=check_plot(scatter_data1,diagram_names1,sfr_mass_folder)
        hist_data1,general_names1=check_plot(hist_data1,general_names1,sfr_mass_folder)

        diagram(scatter_data1,diagram_names1)
        print()
        general(hist_data1,general_names1)
        print()
        var_subplot(hist_data,diagrams)
        return

    diagram(scatter_data,diagram_names)
    print()
    general(hist_data,general_names)
    print()
    var_subplot(hist_data,diagrams)

def step5():
    data_arr=[child_z,
        child_petroMag_u,child_petroMag_g,child_petroMag_r,child_petroMag_i,child_petroMag_z,
        child_h_alpha_flux,child_h_beta_flux,child_oiii_5007_flux,child_nii_6584_flux,
        child_lgm_tot_p50,child_lgm_tot_p16,child_sfr_tot_p84,
        child_absMagU,child_absMagG,child_absMagR,child_absMagI,child_absMagZ]

    name_arr=['child_z',
        'child_petroMag_u','child_petroMag_g','child_petroMag_r','child_petroMag_i','child_petroMag_z',
        'child_h_alpha_flux','child_h_beta_flux','child_oiii_5007_flux','child_nii_6584_flux',
        'child_lgm_tot_p50','child_sfr_tot_p50',
        'child_absMagU','child_absMagG','child_absMagR','child_absMagI','child_absMagZ']

    data=cleaning(data_arr,0)
    print(len(data[0]))
    z_arr=data[0]
    y=data[10]
    e16=data[11]
    ed=[]
    for k1 in range(len(y)):
        ed.append(y[k1]-e16[k1])
    e84=data[12]
    eu=[]
    for k2 in range(len(y)):
        eu.append(e84[k2]-y[k2])
    z_range=z_div(z_arr,Nbins)
    bins_arr=bins_sep(y,z_arr,z_range)
    erru=bins_sep(eu,z_arr,z_range)
    errd=bins_sep(ed,z_arr,z_range)
    x_arr,y_arr=coordinates(z_range,bins_arr)
    ign,err_up=coordinates(z_range,erru)
    ign,err_down=coordinates(z_range,errd)
    plt.figure(1)
    plt.scatter(z_arr,y)
    plt.errorbar(x_arr,y_arr,yerr=[errd,erru],fmt='o',color='r')
    plt.show()

    #plots3(data_arr,name_arr)
    #plotta(data_arr,name_arr)

def step6():
    data_arr=[child_z,
        child_petroMag_u,child_petroMag_g,child_petroMag_r,child_petroMag_i,child_petroMag_z,
        child_h_alpha_flux,child_h_beta_flux,child_oiii_5007_flux,child_nii_6584_flux,
        child_lgm_tot_p50,child_sfr_tot_p50,
        child_absMagU,child_absMagG,child_absMagR,child_absMagI,child_absMagZ]

    name_arr=['child_z',
        'child_petroMag_u','child_petroMag_g','child_petroMag_r','child_petroMag_i','child_petroMag_z',
        'child_h_alpha_flux','child_h_beta_flux','child_oiii_5007_flux','child_nii_6584_flux',
        'child_lgm_tot_p50','child_sfr_tot_p50',
        'child_absMagU','child_absMagG','child_absMagR','child_absMagI','child_absMagZ']

    filename='correlazione.fits'
    v=file_in_folder(filename,out_step_6)
    if overwrite==True or v==0:
        print("Creazione del file '"+filename+"' in corso.")
        data=cleaning(data_arr,0)
        names_arr,vals=general_corr(data,name_arr)
        type_arr,per_arr=corr_type(vals,names_arr)
        corr_fits(names_arr,vals,type_arr,per_arr,filename)
        print("Il file è stato creato con successo.")

    else:
        print("File '"+filename+"' già esistente.")

def run():
    print("Step II: Misurazione della distribuzione statistica di diverse proprietà\n")
    step2()
    print(divisore[:-1])
    print("Step III: Misurazione dell'evoluzione delle proprietà del campione\n")
    step3()
    print(divisore[:-1])
    print("Step IV: Analisi dei diagrammi\n")
    step4()
    #print(divisore[:-1])
    #step5()
    print("Step VI: Studio di eventuali correlazioni\n")
    step6()
run()

elapsed_time=time.time()-start_time
print(time.strftime('%H:%M:%S',time.gmtime(elapsed_time)))
