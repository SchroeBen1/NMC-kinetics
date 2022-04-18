import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import diff
from sklearn import linear_model


#define file path where data is located
TGA_path='#filepath'

#define column names for dataframes
headerlist_1=["Time (min) 2c", "Temp (C) 2c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 2c"]
headerlist_2=["Time (min) 4c", "Temp (C) 4c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 4c"]
headerlist_3=["Time (min) 6c", "Temp (C) 6c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 6c"]
headerlist_4=["Time (min) 8c", "Temp (C) 8c", "Heat Flow (W/g)", "Weight (%)", "Weight (mg)", "alpha 8c"]

#use read_csv to import data in .csv format for each heating rate
#first 1000 columns are dropped as they are noise from TGA and rounds alpha data to 3 decimals
TGA_2ci=pd.read_csv(TGA_path+'#heating rate 1',names=headerlist_1)\
.drop([ 'Heat Flow (W/g)', 'Weight (%)', "Weight (mg)"], axis='columns')
TGA_2ci=TGA_2ci.drop(TGA_2ci.index[0:1000]).astype(float)
TGA_4ci=pd.read_csv(TGA_path+'#heating rate 2',names=headerlist_2)\
.drop([ 'Heat Flow (W/g)', 'Weight (%)', "Weight (mg)"], axis='columns')
TGA_4ci=TGA_4ci.drop(TGA_4ci.index[0:1000]).astype(float)
TGA_6ci=pd.read_csv(TGA_path+'#heating rate 3',names=headerlist_3)\
.drop([ 'Heat Flow (W/g)', 'Weight (%)', "Weight (mg)"], axis='columns')
TGA_6ci=TGA_6ci.drop(TGA_6ci.index[0:1000]).astype(float)
TGA_8ci=pd.read_csv(TGA_path+'#heating rate 4',names=headerlist_4)\
.drop([ 'Heat Flow (W/g)', 'Weight (%)', "Weight (mg)"], axis='columns')
TGA_8ci=TGA_8ci.drop(TGA_8ci.index[0:1000]).astype(float)

#drops all duplicate alpha data and then duplicate time data
TGA_2c=TGA_2ci.drop_duplicates(subset=['alpha 2c'],keep='first',inplace=False)
TGA_2c=TGA_2c.drop_duplicates(subset=['Time (min) 2c'],keep='first',inplace=False)
TGA_4c=TGA_4ci.drop_duplicates(subset=['alpha 4c'],keep='first',inplace=False)
TGA_4c=TGA_4c.drop_duplicates(subset=['Time (min) 4c'],keep='first',inplace=False)
TGA_6c=TGA_6ci.drop_duplicates(subset=['alpha 6c'],keep='first',inplace=False)
TGA_6c=TGA_6c.drop_duplicates(subset=['Time (min) 6c'],keep='first',inplace=False)
TGA_8c=TGA_8ci.drop_duplicates(subset=['alpha 8c'],keep='first',inplace=False)
TGA_8c=TGA_8c.drop_duplicates(subset=['Time (min) 8c'],keep='first',inplace=False)


#time derivative of alpha and its natural log
dadt_2=(diff(TGA_2c['alpha 2c']))/(diff(TGA_2c['Time (min) 2c']))
lnda_2=pd.DataFrame(np.log(dadt_2))
dadt_4=(diff(TGA_4c['alpha 4c']))/(diff(TGA_4c['Time (min) 4c']))
lnda_4=pd.DataFrame(np.log(dadt_4))
dadt_6=(diff(TGA_6c['alpha 6c']))/(diff(TGA_6c['Time (min) 6c']))
lnda_6=pd.DataFrame(np.log(dadt_6))
dadt_8=(diff(TGA_8c['alpha 8c']))/(diff(TGA_8c['Time (min) 8c']))
lnda_8=pd.DataFrame(np.log(dadt_8))

#Finds inverse temperature (Kelvin) as well as remaining NMC concentration from alpha
#makes sure no inf values exist and that arrays are equal dimensions
invT_2=1/(TGA_2c['Temp (C) 2c']+273)
invT_2=invT_2.drop(invT_2.tail(1).index,inplace=False)
nmc_conc2=pd.DataFrame(np.log(1-TGA_2c['alpha 2c'])).replace([np.inf, -np.inf], np.nan, inplace=False) \
    .dropna(subset=['alpha 2c'], how="all")
invT_4=1/(TGA_4c['Temp (C) 4c']+273)
invT_4=invT_4.drop(invT_4.tail(1).index,inplace=False)
nmc_conc4=pd.DataFrame(np.log(1-TGA_4c['alpha 4c'])).replace([np.inf, -np.inf], np.nan, inplace=False) \
    .dropna(subset=['alpha 4c'], how="all")
invT_6=1/(TGA_6c['Temp (C) 6c']+273)
invT_6=invT_6.drop(invT_6.tail(1).index,inplace=False)
nmc_conc6=pd.DataFrame(np.log(1-TGA_6c['alpha 6c'])).replace([np.inf, -np.inf], np.nan, inplace=False) \
    .dropna(subset=['alpha 6c'], how="all")
invT_8=1/(TGA_8c['Temp (C) 8c']+273)
invT_8=invT_8.drop(invT_8.tail(1).index,inplace=False)
nmc_conc8=pd.DataFrame(np.log(1-TGA_8c['alpha 8c'])).replace([np.inf, -np.inf], np.nan, inplace=False) \
    .dropna(subset=['alpha 8c'], how="all")

#defines what feature matrix and target vector for linear regression
#performs multiple linear regression on data
FeatMatrix2=pd.concat([invT_2,nmc_conc2], ignore_index=True, axis=1).dropna(subset=[1], how="all")
lnda_2=lnda_2.head(len(FeatMatrix2))
regr = linear_model.LinearRegression()
coefMatrix2=regr.fit(FeatMatrix2,lnda_2).coef_
CoefIntercept2=regr.fit(FeatMatrix2,lnda_2).intercept_
FeatMatrix4=pd.concat([invT_4,nmc_conc4], ignore_index=True, axis=1).dropna(subset=[1], how="all")
lnda_4=lnda_4.head(len(FeatMatrix4))
coefMatrix4=regr.fit(FeatMatrix4,lnda_4).coef_
CoefIntercept4=regr.fit(FeatMatrix4,lnda_4).intercept_
FeatMatrix6=pd.concat([invT_6,nmc_conc6], ignore_index=True, axis=1).dropna(subset=[1], how="all")
lnda_6=lnda_6.head(len(FeatMatrix6))
coefMatrix6=regr.fit(FeatMatrix6,lnda_6).coef_
CoefIntercept6=regr.fit(FeatMatrix6,lnda_6).intercept_
FeatMatrix8=pd.concat([invT_8,nmc_conc8], ignore_index=True, axis=1).dropna(subset=[1], how="all")
lnda_8=lnda_8.head(len(FeatMatrix8))
coefMatrix8=regr.fit(FeatMatrix8,lnda_8).coef_
CoefIntercept8=regr.fit(FeatMatrix8,lnda_8).intercept_

#calculates activation energy, reaction order, and Arrhenius constant 
#from regression constants
Ea_2=-8.314*(coefMatrix2[:,0])
n2=coefMatrix2[:,1]
A2=np.exp(CoefIntercept2)
Ea_4=-8.314*(coefMatrix4[:,0])
n4=coefMatrix4[:,1]
A4=np.exp(CoefIntercept4)
Ea_6=-8.314*(coefMatrix6[:,0])
n6=coefMatrix6[:,1]
A6=np.exp(CoefIntercept6)
Ea_8=-8.314*(coefMatrix8[:,0])
n8=coefMatrix8[:,1]
A8=np.exp(CoefIntercept8)

#uses calculated kinetic constants to create Arrhenius plot
K2=A2*np.exp((-Ea_2/8.314)*invT_2)
lnK2=np.log(K2)
K4=A4*np.exp((-Ea_4/8.314)*invT_4)
lnK4=np.log(K4)
K6=A6*np.exp((-Ea_6/8.314)*invT_6)
lnK6=np.log(K6)
K8=A8*np.exp((-Ea_8/8.314)*invT_8)
lnK8=np.log(K8)
plt.plot(invT_2, lnK2, 'r-', label="K(T) 2c")
plt.plot(invT_4, lnK4, 'g--', label="K(T) 4c")
plt.plot(invT_6, lnK6, 'b.-',markersize=5,linewidth=1, label="K(T) 6c")
plt.plot(invT_8, lnK8, 'y:', label="K(T) 8c")
plt.title('Arrhenius plot')
plt.xlabel('1/T (K^-1)')
plt.ylabel('ln(K)')
plt.legend()
plt.show()

#formats data and exports it for use in compensation effect analysis
Xc_Ea=[Ea_2, Ea_4, Ea_6, Ea_8]
Yc_lnA=[CoefIntercept2, CoefIntercept4, CoefIntercept6, CoefIntercept8]
Constabledf=pd.concat([pd.DataFrame(Xc_Ea),pd.DataFrame(Yc_lnA)],axis=1)
Constabledf.to_csv('#constable data.csv')
