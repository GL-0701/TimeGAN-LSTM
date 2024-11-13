


import numpy as np
import pandas as pd
from metrics.visualization_metrics import visualization

generated_data = np.load('generated_data.npy')
ori_data = np.load('ori_data.npy')

# 3. Visualization (PCA and tSNE)
# visualization(ori_data, generated_data, 'pca')
# visualization(ori_data, generated_data, 'tsne')

print(generated_data.shape)
data_gan = generated_data.reshape(-1, 10)
#np.savetxt("data1.csv", data1, delimiter=",")
print('Two-dimensional conversion of generated_data: ', data_gan.shape)


print(ori_data.shape)
data_ori = ori_data.reshape(-1, 10)
#np.savetxt("data1.csv", data1, delimiter=",")
print('Two-dimensional conversion of ori_data: ', data_ori.shape)

pd.DataFrame(data_gan).to_csv('generated_data.csv')
pd.DataFrame(data_ori).to_csv('ori_data.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

seq_len=24

sample_size = 300
idx = np.random.permutation(len(ori_data))[:sample_size]

real_sample = np.asarray(ori_data)[idx]
synthetic_sample = np.asarray(generated_data)[idx]

#for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
synth_data_reduced = real_sample.reshape(-1, seq_len)
stock_data_reduced = np.asarray(synthetic_sample).reshape(-1,seq_len)

n_components = 2
pca = PCA(n_components=n_components)
tsne = TSNE(n_components=n_components, n_iter=1200)

#The fit of the methods must be done only using the real sequential data
pca.fit(stock_data_reduced)

pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

fig = plt.figure(constrained_layout=True, figsize=(20,10))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)


#TSNE scatter plot
ax = fig.add_subplot(spec[0,0])
ax.set_title('PCA results', fontsize=20,color='red',pad=10)

#PCA scatter plot
plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:,1].values, c='#08BBBB', alpha=0.8, s=120, label='Original')
plt.scatter(pca_synth.iloc[:,0], pca_synth.iloc[:,1], c='#F6746E', alpha=0.8, s=120, label='Generated')
ax.legend()




ax2 = fig.add_subplot(spec[0,1])
ax2.set_title('TSNE results',fontsize=20,color='red', pad=10)

plt.scatter(tsne_results.iloc[:sample_size*5, 0].values, tsne_results.iloc[:sample_size*5,1].values,c='#08BBBB', alpha=0.8, s=120, label='Original')
plt.scatter(tsne_results.iloc[sample_size*15:,0], tsne_results.iloc[sample_size*15:,1], c='#F6746E', alpha=0.8, s=120, label='Generated')



ax2.legend()

fig.suptitle('Validating synthetic vs real data diversity and distributions',fontsize=16,color='grey')

plt.savefig("PCA-TSNE.png",dpi=1000,bbox_inches = 'tight')

plt.show()
