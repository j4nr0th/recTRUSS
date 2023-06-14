import numpy as np
import pandas as pd

file_loc = "2_not_j/structure1_fullstruct"

connections = pd.read_csv(file_loc + ".con", index_col=0)
radii = []

for i, profile in enumerate(list(connections['profile label'])):
    radius = int(profile.strip('P')) / 1000
    radii.append(radius)

print(np.average(radii))

# print(sum(radius) / len(radius))