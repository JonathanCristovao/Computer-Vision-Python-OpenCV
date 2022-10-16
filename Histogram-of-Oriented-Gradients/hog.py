import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
image = cv2.imread("image.jpg",cv2.COLOR_BGR2GRAY)

fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8,8),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('image original')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 100))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
plt.show()