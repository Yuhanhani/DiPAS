
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


mask_path = 'file path'

mask = np.array(Image.open(mask_path))
mask = mask.astype(np.uint8)
print(np.max(mask), np.min(mask))
mask[mask != 255] = 0
mask[mask == 255] = 1
plt.imshow(mask)
plt.title('original')
plt.show()
mask = mask.astype(np.uint8)

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
sub_path = ('file path')
path = os.path.join(sub_path, 'wind_1_original_{}.png'.format(p_generation))
image.save(path)


# -------pollen transmission----------
for i in range(len(vx_vector)):

    vx = vx_vector[i]
    vy = vy_vector[i]
    if abs(vx)**2 + abs(vy)**2 > 0:
        px = abs(vx)**2 / (abs(vx)**2 + abs(vy)**2)  # probability of transmitting to next x+1 cell
        py = abs(vy)**2 / (abs(vx)**2 + abs(vy)**2)  # probability of transmitting to next y+1 cell
        # print(px, py)
    else:
        continue  # ignore this for loop then (i.e. no wind case)

    # shift in x direction

    index = np.transpose(np.nonzero(mask))  # [row, column]
    mask_shift_x = np.zeros_like(mask)

    if vx >= 0:
        for element in index:
            if element[1] + cell_length <= mask.shape[1]-1:
                mask_shift_x[element[0], element[1] + cell_length] = mask[element[0], element[1]] * px
    if vx < 0:
        for element in index:
            if element[1] - cell_length >= 0:
                mask_shift_x[element[0], element[1] - cell_length] = mask[element[0], element[1]] * px



    # shift in y direction
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


    mask = mask_shift_x + mask_shift_y
    plt.imshow(mask)
    plt.title(f'vx:{vx}, vy:{vy}')
    plt.show()


    # filter based on threshold
    mask_filter = np.copy(mask) # to copy by value rather than by reference
    mask_filter[mask <= personal_threshold] = 0
    mask_filter = mask_filter > 0
    plt.imshow(mask_filter)
    plt.title(f'vx:{vx}, vy:{vy}, (threshold={personal_threshold})')
    plt.show()


    image = Image.fromarray(mask_filter)
    sub_path = ('file path')
    path = os.path.join(sub_path, 'wind_{}_{}_{}.png'.format(i, vx, vy))
    image.save(path)

