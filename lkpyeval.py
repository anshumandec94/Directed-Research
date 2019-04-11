import pandas as pd
from lenskit import batch,topn
from lenskit.metrics import topn as tn
from lenskit.algorithms import als

train =pd.read_csv('ratings20Mfull.csv')
test=pd.read_csv('test_set_all.csv')
train.columns = ['user','item','rating']
test.columns=['user','item','rating']
algo = als.BiasedMF(features=30,iterations=100)
print('Done training the model')
def eval(train,test):
    model=algo.fit(train)
    users=test.user.unique()
    recs=batch.recommend(model, users,20, topn.UnratedCandidates(train))
    print('Recommended to test set of users.')
    return recs

recs=eval(train,test)

#ndcg = recs.groupby('user').rating.apply(tn.ndcg)
#print('Calculated the ndcg. Outputing it to file')
recs.to_csv('recs_20M_ALS_30F_pop_only_user.tsv',sep='\t',header=True,index=False)

