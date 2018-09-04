import numpy as np
import pandas as pd

def get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000, seed=None):
    if train_only:
        X_arr = []
        for _ in range(fold):
            X_perm_arr = []
            perm_idx = np.random.permutation(len(X_train))
            X_train_perm = X_train.loc[X_train.index[perm_idx]]
            y_train_perm = y_train.loc[y_train.index[perm_idx]]

            for j in range(len(X_train_perm)):
                X_perm_arr.append(get_bs_real_mc_mean(
                    col_cat, X_train_perm[:j], y_train_perm[:j], X_valid=X_train_perm[j:j+1], train_only=False, prior=prior
                ))
            X_arr.append(pd.concat(X_perm_arr).loc[X_train.index])
        # take average
        avg = np.zeros(shape=X_arr[0].shape)
        for X_temp in X_arr:
            avg += X_temp.values
        avg /= fold
        real_mc_mean = pd.DataFrame(data=avg, index=X_train.index, columns=[col_cat])
            
        # rand = np.random.rand(len(X_train))
        # lvs = [i / float(fold) for i in range(fold+1)]
        #     msk = (rand >= lvs[i]) & (rand < lvs[i+1])
        #     X_slice = X_train[msk]
        #     X_base = X_train[~msk]
        #     y_base = y_train[~msk]
        #     X_slice = get_bs_real_mc_mean(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
        #     X_arr.append(X_slice)
        # real_mc_mean = pd.concat(X_arr).loc[X_train.index]

    else:
        # merge col_cat with label
        y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
        y_train = y_train.assign(real_mc_mean = y_train['Next_Premium'])

        # get mean of each category and smoothed by global mean
        smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean'].mean()) / (len(x) + prior)
        y_train = y_train.groupby([col_cat]).agg({'real_mc_mean': smooth_mean})
        real_mc_mean = X_valid[col_cat].map(y_train['real_mc_mean'])
        # fill na with global mean
        real_mc_mean = real_mc_mean.where(~pd.isnull(real_mc_mean), np.mean(y_train['real_mc_mean']))

    return(real_mc_mean)

# def get_bs_real_mc_mean(col_cat, X_train, y_train, X_valid=pd.DataFrame(), train_only=True, fold=5, prior=1000):
#     '''
#     In:
#         str(col_cat)
#         DataFrame(X_train),
#         DataFrame(y_train),
#         DataFrame(X_valid),
#         bool(train_only),
#         double(fold),
#     Out:
#         Series(real_mc_prob_distr),
#     Description:
#         get mean of next_premium by col_cat
#     '''
#     if train_only:
#         np.random.seed(1)
#         rand = np.random.rand(len(X_train))
#         lvs = [i / float(fold) for i in range(fold+1)]

#         X_arr = []
#         for i in range(fold):
#             msk = (rand >= lvs[i]) & (rand < lvs[i+1])
#             X_slice = X_train[msk]
#             X_base = X_train[~msk]
#             y_base = y_train[~msk]
#             X_slice = get_bs_real_mc_mean(col_cat, X_base, y_base, X_valid=X_slice, train_only=False, prior=prior)
#             X_arr.append(X_slice)
#         real_mc_mean = pd.concat(X_arr).loc[X_train.index]

#     else:
#         # merge col_cat with label
#         y_train = y_train.merge(X_train[[col_cat]], how='left', left_index=True, right_index=True)
#         y_train = y_train.assign(real_mc_mean = y_train['Next_Premium'])

#         # get mean of each category and smoothed by global mean
#         smooth_mean = lambda x: (x.sum() + prior * y_train['real_mc_mean'].mean()) / (len(x) + prior)
#         y_train = y_train.groupby([col_cat]).agg({'real_mc_mean': smooth_mean})
#         real_mc_mean = X_valid[col_cat].map(y_train['real_mc_mean'])
#         # fill na with global mean
#         real_mc_mean = real_mc_mean.where(~pd.isnull(real_mc_mean), np.mean(y_train['real_mc_mean']))

#     return(real_mc_mean)
