import numpy as np
import pandas as pd

# Search dimensions
# n : number of random samples to draw from each dim
# m : scale of continuous random var
# b : shift of continous random var


# Dropout 
n_dropout = 3
dropout_m = 0.45
dropout_b = 0

# Learning rate
n_lr = 5
lr_m = -2.
lr_b = -3.5

# Whether to learn message-passing features
mp = [True, False]

# Online data augmentation (atomic coordinate perturbations)
n_aug = 4
aug_m = 0.25
aug_b = 0


names = []
name_ind = 0
drops = []
rates = []
passings = []
augs = []
for i in range(n_dropout):
    for j in range(n_lr):
        for k in range(n_aug):
            for passing in mp:
                drop = np.random.rand() * dropout_m + dropout_b
                lr_exp = np.random.rand() * -2. - 3.5
                rate = 1*10**lr_exp
                aug = np.random.rand() * aug_m + aug_b
                
                drops.append(drop)
                rates.append(rate)
                passings.append(passing)
                augs.append(aug)
                names.append(f'rand_search{name_ind}')
                name_ind += 1

settings = {'set_name'   : np.array(names),
            'dropout'    : np.array(drops),
            'lr'         : np.array(rates),
            'mp'         : np.array(passings),
            'online_aug' : np.array(augs),}

settings_df = pd.DataFrame(settings)
print(settings_df.head)

save_name = 'rand_search_settings.pkl'
print(f'Saving proposal settings to {save_name}')
settings_df.to_pickle(save_name)
