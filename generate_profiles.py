import numpy as np
import pandas as pd
import sigfig as sig


def generate_profiles(min_radius, max_radius, steps, filename: str):
    profiles = pd.DataFrame(columns=['profile label', 'radius', 'thickness'])
    # zero_force_radius = [0.001]

    for r in np.geomspace(min_radius, max_radius, steps):
        radius = sig.round(r, sigfigs=2)
        thickness_to_radius = 0.1
        t = sig.round(thickness_to_radius * radius, sigfigs=2)
        label = 'P' + str(radius).replace('.','')
        if len(label) < 5:
            label += ''.join(['0'] * (5 - len(label)))
        profiles = pd.concat([profiles, pd.DataFrame([[label, radius, t]],
                                                     columns=['profile label', 'radius', 'thickness'])])
    profiles.to_csv(filename, index=False)


def smaller_profile(profile_label, fname):
    filename = fname
    if '.' in fname:
        filename = fname.split('.')[0]+'.pro'
    profiles = pd.read_csv(filename)
    current_idx = profiles.index[profiles['profile label'] == profile_label][0]
    if current_idx > 0:
        current_idx = current_idx - 1
    return profiles.iloc[current_idx]['profile label']


def larger_profile(profile_label, fname):
    filename = fname
    if '.' in fname:
        filename = fname.split('.')[0] + '.pro'
    profiles = pd.read_csv(filename)
    current_idx = profiles.index[profiles['profile label'] == profile_label][0]
    if current_idx < len(profiles) - 1:
        current_idx = current_idx + 1
    return profiles.iloc[current_idx]['profile label']


if __name__=='__main__':
    generate_profiles(0.05, 2, 25, "full_structure3/structure1_fullstruct.pro")
