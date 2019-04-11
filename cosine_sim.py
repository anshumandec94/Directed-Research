import pandas as pd
from lenskit import batch,topn
from lenskit.metrics import topn as tn
from lenskit.algorithms import als
import numpy as np
from scipy import spatial
from numpy.linalg import norm
from numpy import dot

trainp=pd.read_csv('ratings20Mpoponlyforsample.csv')
trainf=pd.read_csv('ratings20Mfull.csv')
test=pd.read_csv('test_set_all.csv')

trainp.columns = ['user','item','rating']
trainf.columns = ['user','item','rating']
test.columns=['user','item','rating']

algof=als.BiasedMF(features=30,iterations=100)
algop=als.BiasedMF(features=30,iterations=100)
algof.fit(trainf)
algop.fit(trainp)
full_fmat=algof.user_features_
pop_fmat=algop.user_features_
fullind=algof.user_index_
popind=algop.user_index_
testusers=test['user'].unique().tolist()
user_simscore=pd.DataFrame(columns=['user','simscore'])
r=0
for user in testusers:
    indexf = fullind.get_loc(user)
    indexp = popind.get_loc(user)
    full_v=full_fmat[indexf]
    pop_v=pop_fmat[indexp]
    dot_product=dot(full_v,pop_v)/(norm(full_v)*norm(pop_v))
    user_simscore.loc[r]=[user,dot_product]
    r=r+1
user_simscore.to_csv('ALS_30F_20M_user_simscore.tsv',sep='\t',index=False,header=True)
