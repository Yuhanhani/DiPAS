import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import sys
import os # Because it is not a built-in function, you must always import it. It is a part of the standard library, however, so you will not need to download or install it separately from your Python installation.
import torch
import shapely.geometry
from shapely.geometry import Point, Polygon
import math
import astar_search
import copy

plt.rcParams['figure.figsize'] = [10, 10]

def find_coordinates(fig):

    itemindex = np.where(fig == 255)

    row_coor = itemindex[0]
    column_coor = itemindex[1]

    coor = np.zeros((len(row_coor), 2))

    for i in range(len(row_coor)):
        coor[i, 0] = row_coor[i]
        coor[i, 1] = column_coor[i]

    return coor

#mask_path = '/Users/mirandazheng/Desktop/train_mask/009.png'
#mask_path = '/Users/mirandazheng/Desktop/groundtruth/image1.png'
#mask_path = '/Users/mirandazheng/Desktop/contour/test_19.png'
#mask_path = '/Users/mirandazheng/Desktop/1.15_50epoch/052.png'
#mask_path = '/Users/mirandazheng/Desktop/wind_model/test_b4i5_original.png'
#mask_path = '/Users/mirandazheng/Desktop/3_models/output_final_unet/081.png'
# mask_path = '/Users/mirandazheng/Desktop/output_1.png'
# mask_path = '/Users/mirandazheng/Desktop/output.png'
# mask_path = '/Users/mirandazheng/Desktop/overall_mask.png'
mask_path = '/Users/mirandazheng/Desktop/wind_5_10(0.5)/wind_4_6.1_-2.46.png'


mask = np.array(Image.open(mask_path))
mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
mask = np.pad(mask, pad_width=1, mode='constant', constant_values=1) # to make sure no intersection between boundary and inner grass areas

mask = (mask!=0)  # to ensure a matrix of true and false rather than 0 and 1 hence to plot out contour correctly

print(mask)
print(mask.shape)
plt.imshow(mask)
plt.title('original mask')
plt.show()
mask = mask.astype(np.uint8)
print(type(mask))
print(np.max(mask))
print(np.min(mask))

## Find outer contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # RETR_LIST (show nested contours), RETR_EXTERNAL

print(len(contours)) # 7
# print(contours) # e.g. 3862, 1,2
# a = np.squeeze(contours[2], axis=None )
# print(a.shape) # 3862,2

## Draw
fig = cv2.drawContours(mask, contours, -1, (255, 0, 255), 1)
fig[fig != 255] = 0 # (0,1 for original graph)
fig[fig == 255] = 255
print(type(fig))
np.set_printoptions(threshold=sys.maxsize)
# print(fig[4500:4750, 2000])
print(np.max(fig))
print(np.min(fig))  # fig now has size the same as image, max 255, min 0

plt.imshow(fig)
plt.title('contour')
plt.show()


coor = find_coordinates(fig)
print(coor)
print(coor.shape)
plt.scatter(coor[:,1], mask.shape[0]-coor[:,0]-1)
plt.title('contour')
plt.show()
# plt.scatter([4,2], [2,1])  # must be column against(number of rows - row index - 1) to plot as (x,y) coordinate rather than (row,column) index

plt.scatter(coor[:,0], coor[:,1])
plt.title('contour')
plt.show()


vor = Voronoi(coor)
fig = voronoi_plot_2d(vor)
plt.title(f'Voronoi (initial):{len(vor.ridge_vertices)}')
plt.show()


# clean up by removing lines that cross grass
coordinates_polygon = []
print(type(coordinates_polygon))


for con in contours:


        # print(con.shape)
        contour_index = np.squeeze(con, axis=1) # the middle value
        # print(contour_index.shape)
        contour_index_copy = np.zeros_like(contour_index)
        # print(contour_index)
        contour_index_copy[:, 0], contour_index_copy[:, 1] = contour_index[:, 1], contour_index[:, 0]  # original contours row and column are reverse
        # print(contour_index_copy)
        print(contour_index_copy.shape)
        if contour_index_copy.shape[0] >= 4:
            coordinates_polygon.append(contour_index_copy)
            print('contour added')



print(len(coordinates_polygon)) # list of 7 elements, each is an array of (number of rowsx2) size
# print(coordinates_polygon)
# print(coordinates_polygon[0])
# print(coordinates_polygon[1])
# print(coordinates_polygon[2])

# sys.exit()

print(vor.ridge_vertices)
print(vor.vertices)
print(len(vor.ridge_vertices))
print(len(vor.vertices))

print(f'original:{len(vor.ridge_vertices)}')

# third while loop to deal with outer padded ones and get rid of points connecting dotted lines

element_counter = 0

while element_counter < len(vor.ridge_vertices):

    i = 0
    while i in range(2):  # each ridge_vertex has two elements

        point = vor.vertices[vor.ridge_vertices[element_counter][i]]
        point_to_be_checked = Point(point)

        if vor.ridge_vertices[element_counter][i] == -1: # or (point in coordinates_polygon[-1]):

            vor.ridge_vertices.remove(vor.ridge_vertices[element_counter])
            i = i + 100  # break the current while loop since the element is already removed
            element_counter = element_counter - 1  # to avoid jump over the next element

        i = i + 1

    element_counter = element_counter + 1


print(f'after 3rd while loop:{len(vor.ridge_vertices)}')

fig = voronoi_plot_2d(vor)  # , show_vertices=False, show_points=False)
plt.title(f'Voronoi (after 1st clearance):{len(vor.ridge_vertices)}')
plt.show()


# first while loop to get rid of points that start of end within the polygon (actually can delete this one as the second while loop is more general)
for contour_counter in range(len(coordinates_polygon)-2):  #-2
    print(f'--------contour{contour_counter}------------')
    poly = Polygon(coordinates_polygon[contour_counter])

    print(len(vor.ridge_vertices))

    element_counter = 0

    while element_counter < len(vor.ridge_vertices):
        print(f'contour{contour_counter}:{element_counter}')

        i = 0
        while i in range(2): # each ridge_vertex has two elements

            if vor.ridge_vertices[element_counter][i] != -1:

                point = vor.vertices[vor.ridge_vertices[element_counter][i]]
                point_to_be_checked = Point(point)

                # print(f'ridge_vertex_element:{vor.vertices[vor.ridge_vertices[element_counter][i]]}')
                # print(f'point_to_be_checked.within(poly):{point_to_be_checked.within(poly)}')
                if point_to_be_checked.within(poly):  # or (point in coordinates_polygon[contour_counter]):
                    # print(f'removed:{vor.ridge_vertices[element_counter]}')
                    vor.ridge_vertices.remove(vor.ridge_vertices[element_counter])
                    i = i+100  # break the current while loop since the element is already removed
                    element_counter = element_counter - 1 # to avoid jump over the next element


            i = i+1

        element_counter = element_counter + 1


print(f'after 1st while loop:{len(vor.ridge_vertices)}')
# print(vor.ridge_vertices)
fig = voronoi_plot_2d(vor)
plt.title('Voronoi (after 2nd clearance)')
plt.show()


# second while loop to get rid of points that cross on lie on the boundary of the polygon & the points connect to this
for contour_counter in range(len(coordinates_polygon)-2):  #-2
# for contour_counter in range(1):
    print(f'--------contour{contour_counter}------------')
    poly = Polygon(coordinates_polygon[contour_counter])

    print(len(vor.ridge_vertices))

    element_counter = 0

    while element_counter < len(vor.ridge_vertices):

        if vor.ridge_vertices[element_counter][0] != -1 and vor.ridge_vertices[element_counter][1] != -1:

            point_1 = vor.vertices[vor.ridge_vertices[element_counter][0]]
            point_2 = vor.vertices[vor.ridge_vertices[element_counter][1]]
            line = shapely.geometry.LineString([point_1, point_2])

            # print(f'ridge_vertex_element:{point_1},{point_2}')
            # print(f'point_to_be_checked.within(poly):{line.intersects(poly)}')
            if line.intersects(poly):  # or (point in coordinates_polygon[contour_counter]):
                # print(f'removed:{vor.ridge_vertices[element_counter]}')
                vor.ridge_vertices.remove(vor.ridge_vertices[element_counter])
                element_counter = element_counter - 1 # to avoid jump over the next element


        element_counter = element_counter + 1


print(f'after 2nd while loop:{len(vor.ridge_vertices)}')
# print(vor.ridge_vertices)
fig = voronoi_plot_2d(vor) #, show_vertices=False) #, point_size=1)
plt.title(f'Voronoi (after 2nd clearance):{len(vor.ridge_vertices)}')
plt.show()


# third while loop to deal with outer padded ones and get rid of points connecting dotted lines (has been moved to the first condition check)

# element_counter = 0
#
# while element_counter < len(vor.ridge_vertices):
#
#     i = 0
#     while i in range(2):  # each ridge_vertex has two elements
#
#         point = vor.vertices[vor.ridge_vertices[element_counter][i]]
#         point_to_be_checked = Point(point)
#
#         if vor.ridge_vertices[element_counter][i] == -1: # or (point in coordinates_polygon[-1]):
#
#             vor.ridge_vertices.remove(vor.ridge_vertices[element_counter])
#             i = i + 100  # break the current while loop since the element is already removed
#             element_counter = element_counter - 1  # to avoid jump over the next element
#
#         i = i + 1
#
#     element_counter = element_counter + 1
#
#
# print(f'after 3rd while loop:{len(vor.ridge_vertices)}')
#
# fig = voronoi_plot_2d(vor)  # , show_vertices=False, show_points=False)
# plt.title('Voronoi (after 2nd clearance)')
# plt.show()



# to deal with non-gap padded zeros:

# print(np.transpose(coor))
# coor_transpose = np.transpose(coor)
# if mask.shape[0]-2 in coor_transpose[0,:]: #and or 1 in coor_transpose[0,:] or
#     index = len(coordinates_polygon)-2
#     for i in range(len(coordinates_polygon[index])):


# mask_alternative = np.zeros_like(mask)
# for i in range(len(coordinates_polygon[1])):
#     print(coordinates_polygon[1][i])
#     mask_alternative[coordinates_polygon[1][i][0], coordinates_polygon[1][i][1]] = 1
# print(mask_alternative)
#
# mask_alternative = mask_alternative.astype(np.uint8)
# ## Find outer contours
# contours_alternative, hierarchy_alternative = cv2.findContours(mask_alternative, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#
# coordinates_polygon_alternative = []
#
#
# for con in contours_alternative:
#
#
#         # print(con.shape)
#         contour_index = np.squeeze(con, axis=1) # the middle value
#         # print(contour_index.shape)
#         contour_index_copy = np.zeros_like(contour_index)
#         # print(contour_index)
#         contour_index_copy[:, 0], contour_index_copy[:, 1] = contour_index[:, 1], contour_index[:, 0]  # original contours row and column are reverse
#         # print(contour_index_copy)
#         print(contour_index_copy.shape)
#         if contour_index_copy.shape[0] >= 4:
#             coordinates_polygon_alternative.append(contour_index_copy)
#             print('contour added')
#
# print(coordinates_polygon_alternative)
# Remove overlapped lines that are defined by the same two points first


# new_vor_ridge_vertices = []
#
# for element in vor.ridge_vertices:
#     if element not in new_vor_ridge_vertices:
#         new_vor_ridge_vertices.append(element)
#
# print(new_vor_ridge_vertices)
# print(f'after remove overlapped:{len(new_vor_ridge_vertices)}')



###**************only used in wind case****************************###
# fifth while loop to delete roads that have no gap actually (only used in wind case)
#
# element_counter = 0
#
#
# while element_counter < len(vor.ridge_vertices):
#
#     i = 0
#     while i in range(2):  # each ridge_vertex has two elements
#
#         point = vor.vertices[vor.ridge_vertices[element_counter][i]]
#
#         if point[0] == mask.shape[0] -2 or point[0] == 1 or point[1] == mask.shape[1] -2 or point[1] == 1:
#
#             vor.ridge_vertices.remove(vor.ridge_vertices[element_counter])
#             i = i + 100  # break the current while loop since the element is already removed
#             element_counter = element_counter - 1  # to avoid jump over the next element
#
#         i = i + 1
#
#     element_counter = element_counter + 1
#
#
# print(f'after 5th while loop:{len(vor.ridge_vertices)}')
#
# fig = voronoi_plot_2d(vor)  # , show_vertices=False, show_points=False)
# plt.show()
##**************only used in wind case****************************###





# adjusted version: anchor and removal: 4th while loop to get rid of the remaining middle lines (i.e. not part of closed loops)

four_corner_segment = []
x = mask.shape[0]
y = mask.shape[1]
four_corner_segment = [[0.5,0.5], [0.5, y-1.5], [x-1.5, 0.5], [x-1.5, y-1.5]]
print(four_corner_segment)

vor.ridge_vertices_array = np.array(vor.ridge_vertices)
element_counter = 0
start_point_index = []

while element_counter < len(vor.ridge_vertices):
    i = 0
    while i in range(2):  # each ridge_vertex has two elements

        if np.count_nonzero(vor.ridge_vertices_array == vor.ridge_vertices_array[element_counter][i]) == 1 and (list(vor.vertices[vor.ridge_vertices_array[element_counter][i]]) not in four_corner_segment):

            # print(vor.ridge_vertices_array[element_counter][i], vor.vertices[vor.ridge_vertices_array[element_counter][i]])

            start_point_index.append(element_counter)

            i = i + 100

        i = i + 1

    element_counter = element_counter + 1
print(type(start_point_index))
print(start_point_index)

iteration = 1

while len(start_point_index) != 0:

    for index in sorted(start_point_index, reverse=True):
        if np.count_nonzero(vor.ridge_vertices_array == vor.ridge_vertices[index][0]) == 0 or np.count_nonzero(vor.ridge_vertices_array == vor.ridge_vertices[index][1]) == 0: # already removed (L case)
            print('yes')
            continue
        del vor.ridge_vertices[index]

    # fig = voronoi_plot_2d(vor, show_vertices=False)
    # plt.title(f'Voronoi (3rd clearance : iteration {iteration})')
    # plt.show()

    vor.ridge_vertices_array = np.array(vor.ridge_vertices)
    element_counter = 0
    start_point_index = []

    while element_counter < len(vor.ridge_vertices):
        i = 0
        while i in range(2):  # each ridge_vertex has two elements

            if np.count_nonzero(vor.ridge_vertices_array == vor.ridge_vertices_array[element_counter][i]) == 1 and (
                    list(vor.vertices[vor.ridge_vertices_array[element_counter][i]]) not in four_corner_segment):
                # print(vor.ridge_vertices_array[element_counter][i], vor.vertices[vor.ridge_vertices_array[element_counter][i]])

                start_point_index.append(element_counter)

                i = i + 100

            i = i + 1

        element_counter = element_counter + 1
    print(len(start_point_index))  # normal if always the same number as it may just leave one line to keep tracking and shortened each time geometrically

    iteration = iteration + 1

print(f'after 4th while loop:{len(vor.ridge_vertices)}')
# print(vor.ridge_vertices)
fig = voronoi_plot_2d(vor) #, show_vertices=False)
plt.title(f'Voronoi (after adjusted 3rd clearance):{len(vor.ridge_vertices)}')
plt.show()




# originial version: anchor and track: 4th while loop to get rid of the remaining middle lines (i.e. not part of closed loops)

four_corner_segment = []
x = mask.shape[0]
y = mask.shape[1]
four_corner_segment = [[0.5,0.5], [0.5, y-1.5], [x-1.5, 0.5], [x-1.5, y-1.5]]
print(four_corner_segment)

vor.ridge_vertices_array = np.array(vor.ridge_vertices)
element_counter = 0
start_point_index = []

while element_counter < len(vor.ridge_vertices):
    i = 0
    while i in range(2):  # each ridge_vertex has two elements

        if np.count_nonzero(vor.ridge_vertices_array == vor.ridge_vertices_array[element_counter][i]) == 1 and (list(vor.vertices[vor.ridge_vertices_array[element_counter][i]]) not in four_corner_segment):

            # print(vor.ridge_vertices_array[element_counter][i], vor.vertices[vor.ridge_vertices_array[element_counter][i]])


            if i == 1:
                append_term = [element_counter, 1]  # to make sure in each [a,b], a is the one that starts the single line

            else:
                append_term = [element_counter, 0]

            start_point_index.append(append_term)

            i = i + 100

        i = i + 1

    element_counter = element_counter + 1

print(len(start_point_index))

vor.ridge_vertices_copy = vor.ridge_vertices.copy()  # copy by value rather than copy by reference
counter = 0
for index in start_point_index:

    print(counter)

    # print(f"element:{index}")

    current_pivot_value = vor.ridge_vertices_copy[index[0]][1 - index[1]]
    current_pivot_index = np.where(vor.ridge_vertices_array == vor.ridge_vertices_copy[index[0]][index[1]]) # convert index in l_copy to the index in shortened l
    #print(current_pivot_index)
    # print(len(current_pivot_index[0])) 1 or 0

    if len(current_pivot_index[0]) == 0: # ignore the start if it has been removed in other iterations (L case)
        continue
    current_pivot_index = [current_pivot_index[0][0], current_pivot_index[1][0]]
    #print(current_pivot_index)

    while np.count_nonzero(vor.ridge_vertices_array == current_pivot_value) == 2:  # include self = 2 hence exclude = 1
        vor.ridge_vertices.remove(vor.ridge_vertices[current_pivot_index[0]])
        vor.ridge_vertices_array = np.array(vor.ridge_vertices)
        current_pivot_index = np.where(vor.ridge_vertices_array == current_pivot_value)
        # print(current_pivot_index)
        current_pivot_index = [current_pivot_index[0][0], current_pivot_index[1][0]]
        # print(current_pivot_index)
        current_pivot_value = vor.ridge_vertices_array[current_pivot_index[0], 1 - current_pivot_index[1]]
        # print(current_pivot_value)

    vor.ridge_vertices.remove(vor.ridge_vertices[current_pivot_index[0]])  # remove the last one before closed loop (i.e. last line segment that connects to closed loop)
    # print(current_pivot_index[0])
    vor.ridge_vertices_array = np.array(vor.ridge_vertices)
    counter = counter + 1

print(f'after 4th while loop:{len(vor.ridge_vertices)}')
# print(vor.ridge_vertices)
fig = voronoi_plot_2d(vor) #, show_vertices=False)
plt.title('Voronoi (after 3rd clearance)')
plt.show()

fig = voronoi_plot_2d(vor, show_vertices=False, point_size=0.7)
plt.title(f'Generalised Voronoi:{len(vor.ridge_vertices)}')
plt.show()


# for i in range(len(vor.ridge_vertices)):
#     point_1 = vor.vertices[vor.ridge_vertices[i][0]]
#     point_2 = vor.vertices[vor.ridge_vertices[i][1]]
#     print(vor.ridge_vertices[i], point_1, point_2)

# A* search----------------------------------------------------------------------------------------------------------

def adjacency_dict(vor):

    flat_vor_ridge_vertices = [item for sublist in vor.ridge_vertices for item in sublist]
    adj = {node: [] for node in flat_vor_ridge_vertices} # dict doesn't allow multiple keys hence will keep one if key duplicate
    # print(adj)
    for edge in vor.ridge_vertices:
        node1, node2 = edge[0], edge[1]
        node1_position = vor.vertices[node1]
        node2_position = vor.vertices[node2]
        g_cost = math.sqrt((node1_position[0]-node2_position[0])**2+(node1_position[1]-node2_position[1])**2)  # sum of euclidean distance between each two points
        adj[node1].append([node2, g_cost])
        adj[node2].append([node1, g_cost])
    return adj

adj = adjacency_dict(vor)
# print(adj)
print(len(vor.vertices)) # >= length of dict (as some points inside the polygon have no connection), and number of ridge_vertices is in order of vertices.

def heuristic_dict(vor, goal_index):

    flat_vor_ridge_vertices = [item for sublist in vor.ridge_vertices for item in sublist]
    heu = {node: None for node in flat_vor_ridge_vertices}
    # print(heu)
    for node in flat_vor_ridge_vertices:
        node_position = vor.vertices[node]
        goal_position = vor.vertices[goal_index]
        h_cost = math.sqrt((node_position[0]-goal_position[0])**2+(node_position[1]-goal_position[1])**2) # straightline euclidean distance
        heu[node] = h_cost  # None type doesn't support append
    return heu

def start_goal_finder(input_start_position, input_goal_position, vor):

    flat_vor_ridge_vertices = [item for sublist in vor.ridge_vertices for item in sublist]
    adj = {node: [] for node in flat_vor_ridge_vertices}
    vertices = [list(vor.vertices[key]) for key in adj] # to tear off those points corresponding to already removed lines
    nodes = np.asarray(vertices)
    dist = np.sum((nodes - input_start_position) ** 2, axis=1)
    dist2 = np.sum((nodes - input_goal_position) ** 2, axis=1) # as for loop is ver slow in python, but this use underlying loop in C, which is faster
    index = np.argmin(dist)
    index2 = np.argmin(dist2)
    position = vertices[index]
    position2 = vertices[index2]
    vor_vertices_list = vor.vertices.tolist()
   # print(type(vor_vertices_list))
   # print(vor_vertices_list)
    start_index = vor_vertices_list.index(position)
    goal_index = vor_vertices_list.index(position2)
    return start_index, goal_index

# input_start_position = [4, 4]
# input_goal_position = [10, 10]
# input_start_position = [200, 800]
# input_goal_position = [700, 100]
input_start_position = [4500, 500]
input_goal_position = [2000, 3200]
start_index, goal_index = start_goal_finder(input_start_position, input_goal_position, vor)
print(start_index, goal_index)

heu = heuristic_dict(vor, goal_index)
# print(heu)

fig = voronoi_plot_2d(vor, show_vertices=False)
plt.plot(input_start_position[0], input_start_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
plt.plot(input_goal_position[0], input_goal_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
plt.plot(vor.vertices[start_index][0], vor.vertices[start_index][1], marker="x", markeredgecolor='red')
plt.plot(vor.vertices[goal_index][0], vor.vertices[goal_index][1], marker="x", markeredgecolor='red')
plt.show()

visited_nodes, optimal_nodes = astar_search.AStarSearch(adj, heu, start_index, goal_index)
print('visited nodes: ' + str(visited_nodes))
print('optimal nodes sequence: ' + str(optimal_nodes))

fig = voronoi_plot_2d(vor, show_vertices=False) #, point_size=0.7)
plt.plot(input_start_position[0], input_start_position[1], marker="x", markeredgecolor='red')
plt.plot(input_goal_position[0], input_goal_position[1], marker="x", markeredgecolor='red')
plt.plot(vor.vertices[start_index][0], vor.vertices[start_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
plt.plot(vor.vertices[goal_index][0], vor.vertices[goal_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")

# point_1 = vor.vertices[2313]
# point_2 = vor.vertices[2314]
# point_1 = vor.vertices[12]
# point_2 = vor.vertices[17]
# print(point_1, point_2)
# print(point_1[0], point_1[1])
# plt.plot(point_1[0], point_1[1], 'bx')
# plt.plot(point_2[0], point_2[1], 'bx')

total_path_length = 0
current_0 = vor.vertices[start_index][0]
current_1 = vor.vertices[start_index][1]

for i in optimal_nodes:
    # plt.plot(vor.vertices[i][0], vor.vertices[i][1], marker="x", markeredgecolor='green')
    total_path_length = total_path_length + math.sqrt((current_0-vor.vertices[i][0])**2+(current_1-vor.vertices[i][1])**2)
    plt.plot([current_0, vor.vertices[i][0]], [current_1, vor.vertices[i][1]], 'g-' , linewidth=5)
    current_0 = vor.vertices[i][0]
    current_1 = vor.vertices[i][1]
plt.title(f'shortest_path_length:{total_path_length}')
plt.show()
print(total_path_length)

print(len(vor.ridge_vertices))
print(len(vor.vertices))
print(vor.vertices)

# plot colors on line rather than points to help visualisation

# for i in range(len(optimal_nodes)-1):
#     plt.plot(vor.vertices[goal_index][0], vor.vertices[goal_index][1], marker="o", markeredgecolor='orange',
#             markerfacecolor="orange")


#remove best route to find second best route for visualisation and proof (problem: cannot prove the found route is second best)
# for i in range(len(optimal_nodes)-1):
#     if optimal_nodes[i] < optimal_nodes[i+1]:
#         vor.ridge_vertices.remove([optimal_nodes[i], optimal_nodes[i+1]])
#         # print(optimal_nodes[i], optimal_nodes[i + 1])
#     else:
#         vor.ridge_vertices.remove([optimal_nodes[i+1], optimal_nodes[i]])
#         # print(optimal_nodes[i+1], optimal_nodes[i])
# fig = voronoi_plot_2d(vor, show_vertices=False)
# plt.title('Generalised Voronoi')
# plt.show()



# colored image: row column coordinate with each pixel has width (so seems like having some shift)
# scatter: use x,y coordinate
# voronoi: normal colored image rotate anticlockwise by 90 degrees (with x,y index shown on the graph the same as row column index)
# coordinates polygon use voronoi index (i.e. x,y index but rotated 90 degrees)











###################visualise and calculate sub-optimal routes###########################################################################################################
# vor2 = copy.deepcopy(vor)
# vor2.ridge_vertices.remove([4414, 4415])
# vor2.ridge_vertices.remove([201, 2097])
#
#
# adj = adjacency_dict(vor2)
# heu = heuristic_dict(vor2, goal_index)
# # print(heu)
#
# fig = voronoi_plot_2d(vor2, show_vertices=False)
# plt.plot(input_start_position[0], input_start_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(input_goal_position[0], input_goal_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(vor2.vertices[start_index][0], vor2.vertices[start_index][1], marker="x", markeredgecolor='red')
# plt.plot(vor2.vertices[goal_index][0], vor2.vertices[goal_index][1], marker="x", markeredgecolor='red')
# plt.show()
#
# visited_nodes, optimal_nodes = astar_search.AStarSearch(adj, heu, start_index, goal_index)
# print('visited nodes: ' + str(visited_nodes))
# print('optimal nodes sequence: ' + str(optimal_nodes))
#
# fig = voronoi_plot_2d(vor2, show_vertices=False) #, point_size=0.7)
# plt.plot(input_start_position[0], input_start_position[1], marker="x", markeredgecolor='red')
# plt.plot(input_goal_position[0], input_goal_position[1], marker="x", markeredgecolor='red')
# plt.plot(vor2.vertices[start_index][0], vor2.vertices[start_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(vor2.vertices[goal_index][0], vor2.vertices[goal_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
#
# total_path_length = 0
# current_0 = vor2.vertices[start_index][0]
# current_1 = vor2.vertices[start_index][1]
#
# for i in optimal_nodes:
#     # plt.plot(vor.vertices[i][0], vor.vertices[i][1], marker="x", markeredgecolor='green')
#     total_path_length = total_path_length + math.sqrt((current_0-vor2.vertices[i][0])**2+(current_1-vor.vertices[i][1])**2)
#     plt.plot([current_0, vor2.vertices[i][0]], [current_1, vor2.vertices[i][1]], 'g-' , linewidth=5)
#     current_0 = vor2.vertices[i][0]
#     current_1 = vor2.vertices[i][1]
# plt.title(f'shortest_path_length:{total_path_length}')
# plt.show()
# print(total_path_length)
#
#
# vor2.ridge_vertices.remove([7, 8])
#
# adj = adjacency_dict(vor2)
# heu = heuristic_dict(vor2, goal_index)
# # print(heu)
#
# fig = voronoi_plot_2d(vor2, show_vertices=False)
# plt.plot(input_start_position[0], input_start_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(input_goal_position[0], input_goal_position[1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(vor2.vertices[start_index][0], vor2.vertices[start_index][1], marker="x", markeredgecolor='red')
# plt.plot(vor2.vertices[goal_index][0], vor2.vertices[goal_index][1], marker="x", markeredgecolor='red')
# plt.show()
#
# visited_nodes, optimal_nodes = astar_search.AStarSearch(adj, heu, start_index, goal_index)
# print('visited nodes: ' + str(visited_nodes))
# print('optimal nodes sequence: ' + str(optimal_nodes))
#
# fig = voronoi_plot_2d(vor2, show_vertices=False) #, point_size=0.7)
# plt.plot(input_start_position[0], input_start_position[1], marker="x", markeredgecolor='red')
# plt.plot(input_goal_position[0], input_goal_position[1], marker="x", markeredgecolor='red')
# plt.plot(vor2.vertices[start_index][0], vor2.vertices[start_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
# plt.plot(vor2.vertices[goal_index][0], vor2.vertices[goal_index][1], marker="o", markeredgecolor='orange', markerfacecolor="orange")
#
# total_path_length = 0
# current_0 = vor2.vertices[start_index][0]
# current_1 = vor2.vertices[start_index][1]
#
# for i in optimal_nodes:
#     # plt.plot(vor.vertices[i][0], vor.vertices[i][1], marker="x", markeredgecolor='green')
#     total_path_length = total_path_length + math.sqrt((current_0-vor2.vertices[i][0])**2+(current_1-vor.vertices[i][1])**2)
#     plt.plot([current_0, vor2.vertices[i][0]], [current_1, vor2.vertices[i][1]], 'g-' , linewidth=5)
#     current_0 = vor2.vertices[i][0]
#     current_1 = vor2.vertices[i][1]
# plt.title(f'shortest_path_length:{total_path_length}')
# plt.show()
# print(total_path_length)
#
#
#
#
#
# print(vor.ridge_vertices)
# print(len(vor.ridge_vertices))
#
# print(len(vor2.ridge_vertices))
# print(vor2.ridge_vertices)
















######------------------------------case-study-route-mapping----------------###################
# mask_path = '/Users/mirandazheng/Desktop/toy.png'
# mask = np.array(Image.open(mask_path))
# plt.plot(5.5, 3.5, marker="x", markeredgecolor='red')
# plt.imshow(mask)
# plt.title('original mask')
# plt.show()

# counter = 0
# current_point = 0
# last_point = 0
#
# for i in optimal_nodes:
#     current_point = [vor.vertices[i][1], vor.vertices[i][0]]
#
#     if counter != 0:
#         print(last_point,current_point)
#         plt.plot([last_point[0], current_point[0]], [last_point[1], current_point[1]], 'g-' , linewidth=1)
#
#     #plt.plot(vor.vertices[i][1], vor.vertices[i][0], marker="x", markeredgecolor='red')
#     counter = counter + 1
#     last_point = current_point
#
#
# # plt.imshow(mask)
# # plt.title('original mask')
#
# dir = '/Users/mirandazheng/Desktop/012-a.png'
#
# mask1 = np.array(Image.open(dir))
# mask1 = np.pad(mask1, pad_width=1, mode='constant', constant_values=0)
# mask1 = np.pad(mask1, pad_width=1, mode='constant', constant_values=0)
#
# plt.imshow(mask1)
#
# plt.show()


dir = '/Users/mirandazheng/Desktop/overall-bb.png'

mask1 = np.array(Image.open(dir))
mask1 = np.pad(mask1, pad_width=1, mode='constant', constant_values=0)
mask1 = np.pad(mask1, pad_width=1, mode='constant', constant_values=0)
plt.imshow(mask1)


# find coordinates of drawn lines
contours, hierarchy = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"length of contours:{len(contours)}")

# delete deplicate coordinates from contours
counter = 0
coordinates = []
for contour in contours:

    line_coordinates = set() # Use a set to automatically remove duplicates
    for point in contour:
        x, y = point[0]  # Extract coordinates
        line_coordinates.add((x, y))
    coordinates.append(list(line_coordinates))
    counter = counter + 1

print(coordinates)
print(f"length of coordinates:{len(coordinates)}")


# visualise shortest route
counter = 0
current_point = 0
last_point = 0

for i in optimal_nodes:
    current_point = [vor.vertices[i][1], vor.vertices[i][0]]

    if counter != 0:
        # print(last_point,current_point)
        plt.plot([last_point[0], current_point[0]], [last_point[1], current_point[1]], 'g-' , linewidth=5)

    # plt.plot(vor.vertices[i][1], vor.vertices[i][0], marker="x", markeredgecolor='red')
    counter = counter + 1
    last_point = current_point

# print(mask1)


# visualise pixel-wise drawn routes as points
def find_coordinates(fig):

    itemindex = np.where(fig == 1)

    row_coor = itemindex[0]
    column_coor = itemindex[1]

    coor = np.zeros((len(row_coor), 2))

    for i in range(len(row_coor)):
        coor[i, 0] = column_coor[i]
        coor[i, 1] = row_coor[i]

    return coor  # different from generalised voronoi (column first here)

coor = find_coordinates(mask1)

print(coor)

for i in range(len(coor)):
    plt.plot(coor[i][0],coor[i][1], marker="x", markeredgecolor='red')

# plt.show()


# map shortest route to closest drawn route

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def hausdorff_distance(route1, route2):
    distances1 = [min(euclidean_distance(point1, point2) for point2 in route2) for point1 in route1]
    distances2 = [min(euclidean_distance(point1, point2) for point1 in route1) for point2 in route2]

    return max(max(distances1), max(distances2))


# Example usage
shortest_route=[]
for element in optimal_nodes:
    shortest_route.append((vor.vertices[element][1], vor.vertices[element][0])) ########## changed here to run again
print(shortest_route)   # Coordinates of points on the fake route

route_answer = []
for i in range(len(coordinates)):
    print(coordinates[i])
    distance = hausdorff_distance(shortest_route, coordinates[i])
    print(distance)
    route_answer.append(distance)

print(route_answer)
print(route_answer.index(min(route_answer)))
answer = route_answer.index(min(route_answer))


# draw out the best mapped route

counter = 0
current_point = 0
last_point = 0

for i in coordinates[answer]:

    plt.plot(i[0], i[1], marker="x", markeredgecolor='blue')

image_path = '/Users/mirandazheng/Desktop/overall.png'
image = np.array(Image.open(image_path))
plt.imshow(image)

plt.show()

# --draw on real image--------------------
image_path = '/Users/mirandazheng/Desktop/overall.png'
image = np.array(Image.open(image_path))
plt.imshow(image)

for i in coordinates[answer]:
    plt.plot(i[0], i[1], marker="x", markeredgecolor='blue')

counter = 0
current_point = 0
last_point = 0

for i in optimal_nodes:
    current_point = [vor.vertices[i][1], vor.vertices[i][0]]

    if counter != 0:
        plt.plot([last_point[0], current_point[0]], [last_point[1], current_point[1]], 'r-' , linewidth=5)

    counter = counter + 1
    last_point = current_point

plt.show()

