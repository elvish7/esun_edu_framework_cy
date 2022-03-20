import time
import numpy as np
import matplotlib.pyplot as plt
warp_model = LightFM(loss='warp',
                    learning_schedule='adagrad',
                    max_sampled=100,
                    user_alpha=alpha,
                    item_alpha=alpha)

bpr_model = LightFM(loss='bpr',
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)
warp_duration = []
bpr_duration = []
warp_auc = []
bpr_auc = []

for epoch in range(epochs):
    start = time.time()
    warp_model.fit_partial(interactions, epochs=5)
    warp_duration.append(time.time() - start)
    #warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

for epoch in range(epochs):
    start = time.time()
    bpr_model.fit_partial(interactions, epochs=5)
    bpr_duration.append(time.time() - start)
    #bpr_auc.append(auc_score(bpr_model, test, train_interactions=train).mean())

x = np.arange(epochs)
plt.plot(x, np.array(warp_duration))
plt.plot(x, np.array(bpr_duration))
plt.legend(['WARP loss', 'BPR loss'], loc='upper right')
plt.show()
plt.savefig('loss.png')