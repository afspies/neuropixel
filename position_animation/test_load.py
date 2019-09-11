import scipy.io as sci
boundary_def = sci.loadmat("/home/alex/Desktop/ndc/neuropixel/position_animation/coordinatesData.mat")
# print(boundary_def['coordinatesData'])
print(boundary_def['coordinatesData'][0])
# for box in enumerate([0]):
#     print(box)
#     print(boundary_def.dtype[1][i])