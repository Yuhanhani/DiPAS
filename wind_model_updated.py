# update model and constrain it only to x+1 and y+1
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

cell_length = 350  # 50/100, unit: pixels (although in this model, every quantity is dimensionless)
p_generation = 0.8  # probability of pollen generation
personal_threshold = 0.5  # specific to each patient
vx_vector = [6.10, 6.10, 6.10, 6.10, 6.10]  # horizontal component of wind at each discrete time, unit can be m/s (although in this model, every quantity is dimensionless)
vy_vector = [-2.46, -2.46, -2.46, -2.46, -2.46]  # vertical component of wind at each discrete time, unit can be m/s (although in this model, every quantity is dimensionless)

# vx_vector = [-3.73, -3.73, -3.73, -3.73, -3.73]
# vy_vector = [9.73, 9.73, 9.73, 9.73, 9.73]


#mask_path = '/Users/mirandazheng/Desktop/masks/009.png'
#mask_path = '/Users/mirandazheng/Desktop/groundtruth/image1.png'
#mask_path = '/Users/mirandazheng/Desktop/contour/test_2.png'
#mask_path = '/Users/mirandazheng/Desktop/1.15_50epoch/070.png'
#mask_path = '/Users/mirandazheng/Desktop/1.15_50epoch/batch_4_image_5.png'
#mask_path = '/Users/mirandazheng/Desktop/wind_model/wind_1.png'
mask_path = '/Users/mirandazheng/Desktop/output.png' #008

mask = np.array(Image.open(mask_path))
mask = mask.astype(np.uint8)
# mask = np.array([[0,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]])
#print(mask)
print(np.max(mask), np.min(mask))
mask[mask != 255] = 0 # (0,255 for original graph) only for test image
mask[mask == 255] = 1
# print(mask.shape)
plt.imshow(mask)
plt.title('original')
plt.show()
mask = mask.astype(np.uint8)
# print(type(mask))

# -------pollen generation-----------
mask = mask * p_generation

plt.imshow(mask)
plt.title(f'initial pollen generated, (p_generation={p_generation})')
plt.show()
print(np.max(mask), np.min(mask))

mask_filter = np.copy(mask)
mask_filter[mask <= personal_threshold] = 0
mask_filter = mask_filter > 0

plt.imshow(mask_filter)
plt.title(f'initial pollen generated, (threshold={personal_threshold})')
plt.show()

image = Image.fromarray(mask_filter)
sub_path = ('/Users/mirandazheng/Desktop')
path = os.path.join(sub_path, 'wind_1_original_{}.png'.format(p_generation))
image.save(path)


# -------pollen transmission----------
for i in range(len(vx_vector)):
    #print(mask)

    vx = vx_vector[i]
    vy = vy_vector[i]
    if abs(vx)**2 + abs(vy)**2 > 0:
        px = abs(vx)**2 / (abs(vx)**2 + abs(vy)**2)  # probability of transmitting to next x+1 cell
        py = abs(vy)**2 / (abs(vx)**2 + abs(vy)**2)  # probability of transmitting to next y+1 cell
        # print(px, py)
    else:
        continue  # ignore this for loop then (i.e. no wind case)

    # shift in x direction

    #mask_remain = mask * (1-px)
    index = np.transpose(np.nonzero(mask))  #[row, column]

    mask_shift_x = np.zeros_like(mask)

    if vx >= 0:
        for element in index:
            if element[1] + cell_length <= mask.shape[1]-1:
                mask_shift_x[element[0], element[1] + cell_length] = mask[element[0], element[1]] * px
    if vx < 0:
        for element in index:
            if element[1] - cell_length >= 0:
                mask_shift_x[element[0], element[1] - cell_length] = mask[element[0], element[1]] * px

    # plt.imshow(mask_shift_x)
    # plt.show()


    # shift in y direction
    #mask_remain = mask * (1-py)
    index = np.transpose(np.nonzero(mask))
    mask_shift_y = np.zeros_like(mask)

    if vy >= 0:
        for element in index:
            if element[0] + cell_length <= mask.shape[0]-1:
                mask_shift_y[element[0] + cell_length, element[1]] = mask[element[0], element[1]] * py

    if vy < 0:
        for element in index:
            if element[0] - cell_length >= 0:
                mask_shift_y[element[0] - cell_length, element[1]] = mask[element[0], element[1]] * py


    # plt.imshow(mask_shift_y)
    # plt.show()

    mask = mask_shift_x + mask_shift_y
    plt.imshow(mask)
    plt.title(f'vx:{vx}, vy:{vy}')
    plt.show()
    #print(np.min(mask), np.max(mask))

    # filter based on threshold
    mask_filter = np.copy(mask) # to copy by value rather than by reference
    #print(mask_filter)
    mask_filter[mask <= personal_threshold] = 0
    mask_filter = mask_filter > 0  # convert 1,0 to true,false matrix in order to save it
    #print(mask_filter)
    plt.imshow(mask_filter)
    plt.title(f'vx:{vx}, vy:{vy}, (threshold={personal_threshold})')
    plt.show()
    #print(np.max(mask_filter), np.min(mask_filter))
    image = Image.fromarray(mask_filter)
    sub_path = ('/Users/mirandazheng/Desktop')
    path = os.path.join(sub_path, 'wind_{}_{}_{}.png'.format(i, vx, vy))
    image.save(path)



# mask[mask != 0] = 1 # (0,255 for original graph) only for test image
# mask[mask == 0] = 0

# mask = mask > 0
# plt.imshow(mask)
# plt.show()
# print(mask)
# image = Image.fromarray(mask)
# path = ('/Users/mirandazheng/Desktop/wind_model/test_58.png')
# image.save(path)

