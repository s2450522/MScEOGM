# %%
# Importing relevant libraries
import numpy as np

from skimage import io
from skimage import exposure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# %%
# Helper function for TCI
def composition(Band_01,Band_02,Band_03):
    im_comp=np.dstack([Band_01/Band_01.max(), Band_02/Band_02.max(), Band_03/Band_03.max()])
    for i in range(3):
        v_min, v_max = np.percentile(im_comp[:,:,i],(1,98))
        im_comp[:,:,i] = exposure.rescale_intensity(im_comp[:,:,i],in_range=(v_min,v_max))
    return im_comp

# Read in images
imFolder = "/home/s2450522/_Dissertation/Data/Sentinel-2_2023-01-29/"
coll = io.ImageCollection(imFolder + '*.tiff')

# A list of loaded files
band_names=['Band_01_Clp','Band_02_Clp','Band_03_Clp','Band_04_Clp','Band_05_Clp','Band_06_Clp','Band_07_Clp','Band_08_Clp','Band_08A_Clp','Band_09_Clp','Band_11_Clp','Band_12_Clp']

# %%
# Displaying TCI
im_comp = composition(coll[3],coll[2],coll[1])
plt.figure()
plt.imshow(im_comp)
# plt.show()

# %%
# Transform arrays into numpy stacks
bands=[]
for i in range(len(coll)):
    band = np.asarray(coll[i].data).flatten()
    bands.append(band)

# %%
    # step 2 - stacking band vectors into a table    
X=np.stack(bands).T
print(X.shape)

# %%
# k-means cluster
kmeans = KMeans(n_clusters=10, 
                random_state=2, 
                algorithm="full")
y = kmeans.fit_predict(X)
y.shape 


# %%
# Visualise clusters
y_im=y.reshape(coll[1].shape)
y_im.shape

# %%
# Visualise original and classified
fig, arr = plt.subplots(1,2)
arr[0].imshow(im_comp)
arr[0].set_title("True Colour Composition", size=20)
arr[1].imshow(y_im)
arr[1].set_title("K-means Clustering Output",size=20)

# %%
# Visualise 10 clusters
fig, arr = plt.subplots(5,2)
plt.tight_layout()

arr[0,0].imshow(y_im==0,cmap='gray')
arr[0,0].set_title("cluster 0",size=10)
arr[0,1].imshow(y_im==1,cmap='gray')
arr[0,1].set_title("cluster 1",size=10)
arr[1,0].imshow(y_im==2,cmap='gray')
arr[1,0].set_title("cluster 2",size=10)
arr[1,1].imshow(y_im==3,cmap='gray')
arr[1,1].set_title("cluster 3",size=10)
arr[2,0].imshow(y_im==4,cmap='gray')
arr[2,0].set_title("cluster 4",size=10)
arr[2,1].imshow(y_im==5,cmap='gray')
arr[2,1].set_title("cluster 5",size=10)
arr[3,0].imshow(y_im==6,cmap='gray')
arr[3,0].set_title("cluster 6",size=10)
arr[3,1].imshow(y_im==7,cmap='gray')
arr[3,1].set_title("cluster 7",size=10)
arr[4,0].imshow(y_im==8,cmap='gray')
arr[4,0].set_title("cluster 8",size=10)
arr[4,1].imshow(y_im==9,cmap='gray')
arr[4,1].set_title("cluster 9",size=10)
plt.show()

# %%
# # Assign clusters
y_class=np.zeros(y_im.shape)

y_class[y_im==0]= 4
y_class[y_im==1]= 1
y_class[y_im==2]= 2
y_class[y_im==3]= 4
y_class[y_im==4]= 5
y_class[y_im==5]= 4
y_class[y_im==6]= 6
y_class[y_im==7]= 7
y_class[y_im==8]= 3
y_class[y_im==9]= 7

# %%
#   Visualise original and classified image
# fig, arr = plt.subplots()
plt.tight_layout()

# Visualise original image
plt.figure()
plt.imshow(im_comp)

# Visualise classified image
plt.figure(figsize=(40,20))
plt.imshow(y_class)

# plt.show()

# %%
# Legend
fig = plt.figure()
ax = fig.add_subplot()
plt.imshow(np.hstack((np.zeros((10,200)), np.ones((10,600)),2*np.ones((10,1000)),3*np.ones((10,1400)),4*np.ones((10,1800)),5*np.ones((10,2000)), 6*np.ones((10,2400)))))
ax.text(40, 6, "Water", fontsize=10, color="black")
ax.text(440, 6, "Urban", fontsize=15, color="black")
ax.text(840, 6, "G. Veg", fontsize=15, color="black")
ax.text(1240, 6, "Forest", fontsize=15, color="black")
ax.text(1640, 6, "Bare Soil", fontsize=15, color="black")
ax.text(2040, 6, "Wetland", fontsize=15, color="black")
ax.text(2440, 6, "B. Veg", fontsize=15, color="black")
plt.show()


