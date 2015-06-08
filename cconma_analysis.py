# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:35:46 2015

@author: Stephen

CCONMA CUSTOMER NETWORK CREATION 
PRELIMINARY ANALYSIS

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import os
import pymysql
import netcreate as nc
import sktensor as st
import pickle
import datetime as dt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA, NMF
from time import time
from nimfa import models

#-----------------------------------------------------
#
#
# 1. Subset data and prepare dataframe to build 
#    similarity tensor
#
#
#-----------------------------------------------------

#-----------------------------------------------------
#
# all CUSTOMERS who ordered at least one product
#
#-----------------------------------------------------
con = pymysql.connect(host='localhost', port=3306, user='root', passwd='Rebwooky2008', db='Cconma_db1',charset='utf8')
sql1 = "\
SELECT a.mem_no,  \
a.gender, \
a.f_marriage as marriage, \
datediff(DATE('2014-11-27'),DATE(a.member_birth))/365 AS age \
FROM cconma_member AS a \
JOIN cconma_order AS b \
ON a.mem_no = b.mem_no \
GROUP BY mem_no \
HAVING Count(b.ocode) > 0 \
;"
df1 = psql.read_sql(sql1,con)
con.close()

# make int category for age where NA is 10000 for distance computation
df1.age = df1.age.replace(float('nan'), np.nan)

# categorize ages to make integer category replacement
df1['agecat'] = 'NA'
df1.loc[df1.loc[:,'age']<100,'agecat'] = '55_'
df1.loc[df1.loc[:,'age']<55,'agecat'] = '50_55'
df1.loc[df1.loc[:,'age']<50,'agecat'] = '45_50'
df1.loc[df1.loc[:,'age']<45,'agecat'] = '40_45'
df1.loc[df1.loc[:,'age']<40,'agecat'] = '_40'

# RECODING FACTORS TO INTEGERS
# GENDER:      M : 0,  W : 1,    MISSING: np.NaN
# MARRIAGE:    N : 0,  Y : 1,    MISSING: np.NaN
# AGE:  categories shown above
df1['genderint'] = df1.gender.replace(["M","W",None],[0,1,np.nan])
df1['marriageint'] = df1.marriage.replace(["N","Y",None],[0,1,np.nan])
df1['agecatint'] = df1.agecat.replace(['_40','40_45','45_50','50_55','55_','NA'],
                                      [0,1,2,3,4,np.nan])

df1.drop(labels=['agecat'],axis=1,inplace=True)




#-------------------------------------------------
#
# all PRODUCT REVIEWS
#
#-------------------------------------------------
con = pymysql.connect(host='localhost', port=3306, user='root', passwd='Rebwooky2008', db='Cconma_db1',charset='utf8')
sql2 = "\
SELECT mem_no, \
pcode, \
point \
FROM cconma_productreview \
;"
df2 = psql.read_sql(sql2,con)
con.close()

df2['pref'] = df2.pcode.str.split("-").apply(lambda x: x[0])


#df2piv = df2.pivot_table(index=['mem_no'],columns='pcode',values='point',
#                         aggfunc=len,fill_value='NA').reset_index(drop=False)
#np.shape(df2piv)

#-------------------------------------------------
#
# all ORDERS JOIN MEM_NO
#
#-------------------------------------------------

#con = pymysql.connect(host='localhost', port=3306, user='root', passwd='Rebwooky2008', db='Cconma_db1',charset='utf8')
#sql3 = "\
#SELECT a.mem_no,  \
#date(a.order_date) as odate,  \
#b.pcode,  \
#b.order_number as qty, \
#b.order_price as price  \
#FROM cconma_order as a  \
#JOIN cconma_order_product as b  \
#ON a.ocode = b.ocode;  \
#"
#df3 = psql.read_sql(sql3,con)
#con.close()
#
#df3.to_csv("memno_pcode_ocode_qty_join.csv", index=False)

# faster to load from CSV avoid slow SQL join
#dtypes = {'mem_no':np.int32, 'odate':dt.datetime(), 'pcode':object, 'qty':np.int32}
#names = ['mem_no','odate','pcode','qty']
df3 = pd.read_csv("memno_pcode_ocode_qty_join.csv", sep=",")
df3['odate'] = pd.to_datetime(df3.odate, format='%Y-%m-%d')

df3.drop(labels=['ocode','price'], axis=1, inplace=True)

#-------------------------------------------------
##make pcode prefix variable
#-------------------------------------------------
# split at "-" and keep only first element in split list per row
df3['pref'] = df3.pcode.str.split("-").apply(lambda x: x[0])
cols = list(df3)
cols.insert(0, cols.pop(len(cols)-1))
df3 = df3[cols]
df3.head()

#----------------------------------------------------
# # order FREQ by CUSTOMER, PRODUCTPREFIX
#----------------------------------------------------

# checking orders by date to make train/test split by time
df3.loc[df3.odate < '2014-06-01', :].shape[0]
df3.loc[df3.odate >= '2014-06-01', :].shape[0]




pcf = df3.groupby(['mem_no','pref']).agg({'qty':sum})
pcf.reset_index(inplace=True)
pcf.head()

## look at log quantity distribution
#pc_count_distr = np.array([pcf.loc[pcf.qty > i].shape[0] for i in np.arange(pcf.qty.max())])
#plt.hist(np.log(pc_count_distr))
# customer product order frequency
pcfsub = pcf.loc[pcf.qty > 2,:]

#---------------------------------------------------
# Make train and test subsets by date
#_-------------------------------------------------
df3agg = df3.groupby(['mem_no','pref']).agg({'qty':sum})
df3agg.reset_index(inplace=True)

# training set 2012 January to  2014 June
df3train = df3.loc[df3.odate < dt.date(2014,6,1), :]
df3train.head()

# test set: 2014 June - October
df3test = df3.loc[df3.odate >= dt.date(2014,6,1), :]
df3test.head()

# check the test product not in training set products
testprod = df3test.pref.unique()
trainprod = df3train.pref.unique()
testprodmatch = [prod for prod in testprod if prod in trainprod]
# 545 products purchased after June 1, 2014 to be used for
# model evaluation
# reduce training set to only products included in testing set
df3train = df3train.loc[df3train.pref.isin(testprodmatch),:]


# order freq by customers
pcftrain = df3train.groupby(['mem_no','pref']).agg({'qty':sum})
pcftrain.reset_index(inplace=True)
pcftrain.head()

# order freq by customers
pcftest = df3test.groupby(['mem_no','pref']).agg({'qty':sum})
pcftest.reset_index(inplace=True)
pcftest.head()



#----------------------------------------------------#
#
# subset of n most commonly reviewed products
#
#----------------------------------------------------

         # here using all products
nprods = len(df2.pcode.unique())

pcodecount = df2.groupby(['pcode']).agg({'pcode':len}).rename(columns={'pcode':'freq'})
pcodecount.reset_index(inplace=True)
#pccsub = pcodecount.loc[pcodecount.loc[:,'freq'] >= 10,:]
pcodecount.sort(columns='freq', axis=0, ascending=False, inplace=True)
pcodecount.head()
# subset of top 3 most reviewed products
pcodesub = pcodecount.iloc[0:nprods,:]

#----------------------------------------------------
#
#
#----------------------------------------------------

#-----------------------------------------------------
# find unique pcodes after before the hyphen "-"
#_----------------------------------------------------

up = pcodecount.pcode.unique()
pref = np.unique([ up[x].split("-")[0] for x in np.arange(len(up)) ])
len(pref)
# only 204 unique prefixes


#mnu = pcf.mem_no.unique()
#pcf.loc[  pcf.mem_no == mnu[16] and pref[1] in pcf.pcode[1] , : ]
#
## for MEMBER mnu[x] 
## if PRODUCTPREFIX in PURCHASES 
#index1 = [ pcf.loc[pcf.mem_no==mnu[0]].pcode[i].startswith(pref[1]) for i in np.arange(pcf.pcode.shape[0])]
#
## PRODUCTS by MEMBER
#pcf.loc[pcf.mem_no==mnu[0]]

# 

# TOO SLOW FOR LOOP
#cleaned_list = []
#for i in np.arange(pcf.pcode.shape[0]):
#    try:
#        pcf.loc[pcf.mem_no==mnu[0]].pcode[i].startswith(pref[1])
#    except KeyError:
#        pass
#    else:
#        cleaned_list.append(i)
#    if i % 50 == 0:
#        print(str(i))

#---------------------------------------------------
#
# order data by customer for similarity tensor
# subset by most common products ( > 3 orders)
#
#---------------------------------------------------

prodQty = 1

df3.head()
df3long = df3.groupby(['mem_no','pref']).agg({'qty':sum}).reset_index(inplace=False)
df3sub = df3long.loc[ (df3long.qty > prodQty), :].copy()

df3sub['pref'] = df3sub.pref.apply(lambda x: 'QTY_%s' % (x))



# CAST WIDE 
# watch memory
df3wide = df3sub.pivot_table(index=['mem_no'],columns='pref',values='qty',
                         aggfunc=np.mean, fill_value=np.nan).reset_index(drop=False)


#-----------------------

#df3long.to_csv("order_count_memno_prod_long.csv", delimiter=",", index=False)

M = df3wide.shape[1]
#df3widenorm = df3wide.copy()
#df3widenorm.iloc[:,1:M] = (df3wide.iloc[:,1:M] - df3wide.iloc[:,1:M].mean()) / (df3wide.iloc[:,1:M].max() - df3wide.iloc[:,1:M].min())
#df3widenorm2 = df3wide.copy()
#df3widenorm2.iloc[:,1:M] = (df3wide.iloc[:,1:M] - df3wide.iloc[:,1:M].mean()) / df3wide.iloc[:,1:M].std() 
summemqty = df3wide.sum(axis=1).reset_index(inplace=False).rename(columns={'index':'mem_no',0:'freq'})
summemqty.sort(columns='freq', ascending=False)

# find similar neighborhood of product quantity orders by taking logs and rounding
df3widesub = df3wide.loc[df3wide.mem_no.isin(summemqty.mem_no),:]
df3widesublog = df3widesub.copy()
df3widesublog.iloc[:,1:M] = np.log(df3widesub.iloc[:,1:M])
M = df3widesublog.shape[1]
df3widesublog.iloc[:,1:M] = np.round(df3widesublog.iloc[:,1:M],0)
#for i in np.arange(df3widesublog.shape[0]):
#    for j in np.arange(1,df3widesublog.shape[1]):
#        df3widesublog.iloc[i,j] = np.int8(round(df3widesublog.iloc[i,j],0))
#    print('\n completed member %s' % (i))

#
## RENAME COLUMNS ???
#renamecols = {col:str('QTY_' + col) for col in df3widesublog.iloc[:,1:M].columns}
#
#renamecolslist = df3widesublog.columns
#renamecolslistexclude1 = [ str('QTY_' + col) for col in df3widesublog.iloc[:,1:M].columns]
#renamecolslist[1:M] - renamecolslistexclude1
#
#df3widesublog.iloc[:,1:M].rename(columns=renamecols, inplace=True)

#------------------------------------------------------
#
# subset of reviews by only the customers who reviewed the most commonly reviewed products
#
#----------------------------------------------------
# subset of reviews including only most reviewed products
reviewsub = df2.loc[ df2['pcode'].isin(list(pcodesub.pcode)) , : ]

#------------------------------------------------------
# subset of customers who reviwed products (in the nprod subset above)
#----------------------------------------------------
memsub = df1.loc[df1['mem_no'].isin(list(reviewsub.mem_no)) , : ]
# remove member 0
memsub = memsub.loc[~memsub['mem_no'].isin([0]) , :]
memsub.drop(labels=['gender','marriage','age'],axis=1, inplace=True)

# make reviewed product into dataframe column
prodsub = reviewsub.pivot_table(index=['mem_no'],columns='pref',values='point',
                         aggfunc=np.mean, fill_value="NA").reset_index(drop=False)
prodsub = prodsub.replace("NA",np.nan)

# Round PRODUCT REVIEW MEAN to int
prodsub2 = prodsub.copy()
M = prodsub2.shape[1]
prodsub2[ ( prodsub2.iloc[:,1:M] > 0)    & ( prodsub2.iloc[:,1:M] < 1.5 )   ] = 1
prodsub2[ ( prodsub2.iloc[:,1:M] >= 1.5) & ( prodsub2.iloc[:,1:M] < 2.5 )   ] = 2
prodsub2[ ( prodsub2.iloc[:,1:M] >= 2.5) & ( prodsub2.iloc[:,1:M] < 3.5 )   ] = 3
prodsub2[ ( prodsub2.iloc[:,1:M] >= 3.5) & ( prodsub2.iloc[:,1:M] < 4.5 )   ] = 4
prodsub2[ ( prodsub2.iloc[:,1:M] >= 4.5)                                    ] = 5

#prodpca = PCA(n_components=3)
#prodpca.fit(prodsub2.iloc[:,1:prodsub2.shape[1]])

#-------------------------------------------------------
# data subset to build similarity tensor
#------------------------------------------------------
dfsub = memsub.join(prodsub2.iloc[:,1:np.shape(prodsub2)[1]], on='mem_no', how='left')
#dfsub = dfsub.drop(labels=['age'], axis=1)
dfsub.head()

#
# PICK UP HERE
# HOW TO COMBINE REVIEWS AND PRODUCT QTY SINCE ALL USE SAME PREF ?????
#
dfsub = dfsub.join(df3widesublog.iloc[:,1:df3widesublog.shape[1]], on='mem_no', how='left')

##-------------------------------------------------------
## Replace numeric rating with category high=[4,5] low=[1,2,3]
##------------------------------------------------------
#dfsub.iloc[:,1:dfsub.shape[1]].replace(to_replace=[4,5], value='high', inplace=True  )
#dfsub.iloc[:,1:dfsub.shape[1]].replace(to_replace=[1,2,3], value='low', inplace=True  )
#dfsub.iloc[:,1:dfsub.shape[1]].replace(to_replace=np.NaN, value='NA', inplace=True  )


#-------------------------------------------------------
#
# Dimensionality reduction on main regression behavior predictors
# PCA
#
#------------------------------------------------------

#-------------------------------------------
# QUANTITY PCA USING PREF TOTAL (not individual product)
#------------------------------------------
# narrow down products included to those most frequently bought (> 3 times)
prodQty = 2

df3.head()
df3long = df3.groupby(['mem_no','pref']).agg({'qty':sum}).reset_index(inplace=False)
df3sub = df3long.loc[ (df3long.qty > prodQty), :].copy()

# orders PCA
df3decomp = df3sub.pivot_table(index=['mem_no'],columns='pref',values='qty',
                         aggfunc=np.mean, fill_value=0).reset_index(drop=False)

nComps = 10
N, M = df3decomp.shape
X = df3decomp
qtypca = PCA(n_components = nComps)
qtypca.fit(X)
plt.figure()
plt.plot(np.arange(1,nComps+1),qtypca.explained_variance_ratio_, marker='^')
plt.title('Purchases Quantity: Explained Variance')
plt.xlabel('Principle Components')
plt.savefig("qty_pca_10.png",dpi=200)

qtytrans = qtypca.transform(X)
dfpcaqty = pd.DataFrame(qtytrans)
dfpcaqty = pd.concat( (df3decomp.iloc[:,0], dfpcaqty), axis=1)
dfpcaqty = dfpcaqty.iloc[1:N,:]
dfpcaqty.columns = ['mem_no','qtyPC0','qtyPC1','qtyPC2','qtyPC3']

#----------------------------------------------
# REVIEWS PCA 
# USING INDIVIDUAL PRODUCT REVIEWS (not pref)
#----------------------------------------------

revCount = 4

df2long = df2.groupby(['mem_no','pref']).agg({'point':np.mean}).reset_index(inplace=False)
#df2count = df2.groupby(['pref']).agg({'point':len}).reset_index(inplace=False)
#df2subprefs = df2count.loc[df2count.point > revCount,:]
#df2sub = df2long.loc[ df2long.pref.isin(df2subprefs.pref), :].copy()
#df2sub.shape

revdecomp = df2.pivot_table(index=['mem_no'],columns='pref',values='point',
                         aggfunc=len,fill_value=0).reset_index(drop=False)

nComps = 10
N, M = revdecomp.shape
X = revdecomp.iloc[:,1:M]
revpca = PCA(n_components = nComps)
revpca.fit(X)
plt.figure()
plt.plot(np.arange(1,nComps+1),revpca.explained_variance_ratio_, marker='^')
plt.title('Product Reviews: Explained Variance')
plt.xlabel('Principle Components')
plt.savefig("qty_pca_10.png",dpi=200)

revtrans = revpca.transform(X)
dfpcarev = pd.DataFrame(revtrans)
dfpcarev = pd.concat( (revdecomp.iloc[:,0], dfpcarev), axis=1)
dfpcarev = dfpcarev.iloc[1:N,:]
dfpcarev.columns = ['mem_no','revPC0','revPC1','revPC2','revPC3']

# Combine 
dfpca = dfpcarev.join(dfpcaqty.iloc[:,1:dfpcaqty.shape[1]], on='mem_no', how='inner')
dfreg = df1[['mem_no','age','gender', 'marriage' ]].join(dfpca.iloc[:,1:dfpca.shape[1]], on='mem_no', how='inner')
dfreg = df3agg.loc[df3agg.mem_no != 0,:].join(dfreg.iloc[:,1:dfreg.shape[1]], on='mem_no', how='inner')

dfreg.to_csv("reg_df_qty_rev_demog_pca.csv",delimiter=",",index=False)

#----------------------------------------------------
#
#
# 2. build sim tensor from data subset in step 1
#    decompose using RESCAL_ALS
#    then create network and plot
#
#
#_---------------------------------------------------

#def build_sim_tensor(x, offset=0, Pandas=True, matchValue=1, Sim=True, metric='hamming', *args, **kwargs):
#    """build an r-mode similarity adjacency tensor;
#    Inputs: offset is number of columns in df to skip (shape(df)[1] =
#    offset + r)
#    df columns [offset:ncol] are the r relationship types;
#    output: list of csr_matrix sparse matrices  
#    (format for RESCAL_ALS input)
#    """
#    time0 = time()
#    X = []
#    for j in np.arange(offset,x.shape[1]):
#        if Pandas:
#            if Sim:
#                simmat = 1 - squareform(pdist(np.asarray(x.iloc[:,(j-1):j]), metric) )
#            else:
#                simmat = squareform(pdist(np.asarray(x.iloc[:,(j-1):j]), metric) )
#        else:
#            if Sim:
#                simmat = 1 - squareform(pdist(np.asarray(x[:,(j-1):j]), metric) )
#            else:
#                simmat = squareform(pdist(np.asarray(x[:,(j-1):j]), metric) )
#        simmat[np.triu_indices(simmat.shape[0],k=0)] = None
#        indices = np.where(simmat==matchValue)
#        n = x.shape[0]
#        holder = st.csr_matrix( ( np.ones(indices[0].shape[0]), indices ),
#                                  shape=(n,n), dtype=np.int8)
#        X.append(holder)
#        if j % np.ceil((x.shape[1]-offset)/10) == 0 :
#                print('\ncompleted feature %s' % ( j ) ) 
#    
#    timeout = time() - time0    
#    if timeout <= 120:
#        print('\nTensor build elapsed time: %s seconds' % ( round(timeout,3) ))    
#    elif timeout <= 3600: 
#        print('\nTensor build elapsed time: %s minutes' % ( round(timeout/60,3) ))
#    else:
#        print('\nTensor build elapsed time: %s hours' % ( round(timeout/3600,3) ))
#    
#    return X
    

#----------------------------------------------
#
# 2.A  Create new netCreate objects by BUILDING TENSOR
#
#----------------------------------------------

# 1. create object without premade similarity tensor
b = nc.netCreate()
print(b)

# 2. build the similarity tensor from the data subset

#nmem = 3000
#np.random.seed(111)
#sample = np.random.choice(np.arange(dfsub.shape[0]), nmem, replace=False)

# match members from regression dataframe to members in network tie prediction
memindex = dfreg.mem_no.unique()
# check combatibility with df3wide for latter creation of SIF
memindex = [ x for x in memindex if x in df3wide.mem_no ]
dfinput = dfsub.loc[ dfsub.mem_no.isin(memindex) , : ].copy()
dfinput = dfinput.loc[dfinput.mem_no.isin(df3wide.mem_no) , :].copy()
# bulid sim tensor
b.build_sim_tensor( dfinput, offset=1)
print(b)

## reuse same similarity tensor build in object b
c = nc.netCreate(b.X)

import gc
gc.collect()
#-----------------------------------------------
#
# 2.B  reload already saved TENSOR and use to create netCreate object
#
#-----------------------------------------------

#b = pickle.load( open( "netcreate_n3_obj_high_rank_low_reg.p", "rb" ) )
#c = pickle.load( open( "netcreate_n3_obj_low_rank_high_reg.p", "rb" ) )


#-------------------------------------
## Compute R, A, AAT, AATnn by decomposing tensor X using RESCAL_ALS
#-------------------------------------

#--------------- HIGH RANK---LOW REGULATION ---------------------------
rank = int( b.X[0].shape[0]* 0.99 )   # rank ~ 95% of number of people
reg = 5

#time0 = time()
# decompose tensor
b.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg)


#averge degree of k = 5
minEdges = int(b.X[0].shape[0]*7)

## create network via sampling methods specified
b.net_create(minEdges=minEdges, deterministic=True, Bernoulli=True, 
             compute_fit=False, plotting=True, color_threshold=0.45)
#print(str(round(time()-time0,3)))



#--------------- LOW RANK---HIGH REGULATION ----------------------------
rank = int(np.ceil(c.X[0].shape[0]* 0.7 ))   # rank ~ 90% of number of people
reg = 20

# decompose tensor
c.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg)

## create network via sampling methods specified
c.net_create(minEdges=minEdges, deterministic=True, Bernoulli=True, 
             compute_fit=False, plotting=True, color_threshold=0.45)




# save netCreate object
pickle.dump( b , open("nc_regpred_n670_obj_high_rank95_low_reg4_colthresh45.p","wb" ) )
pickle.dump( c , open("nc_regpred_n670_obj_low_rank70_high_reg15_colthresh45.p","wb" ) )




##plot the three different probability distributions
df = pd.DataFrame({'HrankLreg':b.pred_rank['Bernoulli'][['prob']].reset_index()['prob'],
                   'LrankHreg':c.pred_rank['Bernoulli'][['prob']].reset_index()['prob'] })

me = int( np.ceil(c.X[0].shape[0] * (c.X[0].shape[0] - 1) / 50 ) )
df.iloc[:,:].sort().plot(marker='^',markevery=me,title="Network 'Next Tie Probability'\nby tensor decomp hyperparams")




#-----------------------------------------------------
#
# Make Social Influence Factor Matrix
# Ties strengths by other members' purchases
#
#_----------------------------------------------------

# MISSING 5 PEOPLE FROM INPUT TO OUTPUT ????????
df3widesub = df3wide.loc[df3wide.mem_no.isin(dfinput.mem_no),:]

# replace all df3 np.NaN's with 0's for dot product
df3widesub = df3widesub.replace(np.nan, 0, inplace=False).copy()

# compute the weights matrix
W = b.SIF.dot( df3widesub.iloc[:,1:df3widesub.shape[1]] ) 

# to long form
W = np.hstack((dfinput.mem_no[:,None], W))

# back to pandas dataframe and add the mem_no to join with the regression data
Wdf = pd.DataFrame(W)
Wdf.columns = df3widesub.columns
Wlong = pd.melt(Wdf, id_vars=['mem_no'])
Wlong.rename(columns={'value':'netWeight'}, inplace=True)

# merge with regression data frame
cols = dfreg.columns
cols = [col for col in cols if col not in ['mem_no']]
Wlong['pref'] = Wlong.pref.apply(lambda x: x.split("_")[1])
#dfregall = Wlong.loc[:,['mem_no','netWeight']].join(dfreg.loc[:,cols], on='mem_no',how='inner',lsuffix='_l',rsuffix='_r')

Wlong.to_csv("Wlong.csv",delimiter=",",index=False)















