#!/usr/bin/env python
# coding: utf-8

# In[3]:


# pip install opencv-python


# In[1]:


import cv2
import numpy as np


# In[7]:


cameraman = cv2.imread("C:\\Users\\syama\\Desktop\\fip3.jpg",cv2.IMREAD_GRAYSCALE)
text = cv2.imread("C:\\Users\\syama\\Desktop\\fip-4.png",cv2.IMREAD_GRAYSCALE)


# In[8]:


text_resized = cv2.resize(text, (cameraman.shape[1], cameraman.shape[0]))


# In[9]:


alpha = 0.5  # Adjust transparency of text
beta = (1.0 - alpha)
superimposed_image = cv2.addWeighted(cameraman, alpha, text_resized, beta, 0.0)


# In[10]:


_, thresholded_image = cv2.threshold(superimposed_image, 200, 255, cv2.THRESH_BINARY)


# In[11]:


cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[6]:





# In[7]:





# In[12]:


# Convert to double
text_double = text.astype(np.float64)


# In[13]:


# Invert the text image
text_inverted = cv2.bitwise_not(text)


# In[14]:


text_inverted_double = text_inverted.astype(np.float64)


# In[15]:


m_double = text_double * text_inverted_double


# In[16]:


m = np.uint8(m_double)


# In[17]:


cv2.imshow('Image m', m)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(superimposed_image, cmap='gray')
axes[0].set_title('Superimposed Image')
axes[0].axis('off')


axes[1].imshow(m, cmap='gray')
axes[1].set_title('Image m')
axes[1].axis('off')


plt.tight_layout()
plt.show()


# ###### The first approach directly superimposes the text onto the cameraman image and then applies thresholding, while the second approach defines a new image 'm'  based on the given mathematical expression involving the original text image and its negation. The second approach does not directly involve the cameraman image and operates solely on the text image.

# In[ ]:




#c
# Gray Scale image of an image

# In[2]:


import cv2
import numpy as np


# In[4]:


image = cv2.imread('C:\\Users\\syama\\Desktop\\mandrill.mat.jpg')


# In[5]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[6]:


filter_sizes = [3, 7, 11, 21]
std_deviations = [(0.5, 1, 2), (1, 3, 6), (1, 4, 8), (1, 5, 10)]


# In[ ]:





# In[10]:





# In[ ]:





# ###### This code iterates through each combination of filter size and standard deviation and applies the Gaussian filter to the grayscale image. It displays the filtered image for each combination.

# In[ ]:





# In[18]:



fig, axes = plt.subplots(len(filter_sizes), len(std_deviations), figsize=(15, 15))

best_score = float('inf')
best_gaussian = None
best_size = None
best_std = None

# Applying Gaussian filters with different sizes and standard deviations
for i, size in enumerate(filter_sizes):
    for j, stds in enumerate(std_deviations):
        for std in stds:
            gaussian_image = cv2.GaussianBlur(gray_image, (size, size), std)
            score = np.sum(np.abs(gray_image.astype(np.float32) - gaussian_image.astype(np.float32)))
            
            # Updating best filter if the current one has a lower score
            if score < best_score:
                best_score = score
                best_gaussian = gaussian_image
                best_size = size
                best_std = std
            axes[i, j].imshow(gaussian_image, cmap='gray')
            axes[i, j].set_title(f'Size: {size}, Std: {std}')
            axes[i, j].axis('off')
plt.tight_layout()
plt.show()


# ###### As the standard deviation increases, the smoothing effect of the Gaussian filter becomes more pronounced.The whiskers may start to disappear or become less distinct as the standard deviation increases, depending on the specifics of the image and the whiskers themselves.

# In[19]:


print(f"Best Gaussian Filter - Size: {best_size}, Std: {best_std}")
cv2.imshow('Best Gaussian Filter', best_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




