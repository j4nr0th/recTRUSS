import numpy as np
import pandas as pd


def generate_profiles(min_radius, max_radius, steps):
    profiles = pd.DataFrame(columns=['profile label', 'radius', 'thickness'])
    for r in np.linspace(min_radius, max_radius, steps):
        thickness_to_radius = 0.1
        t = thickness_to_radius * r
        label = 'P' + str(round(r,3)).replace('.','')
        if len(label) < 3:
            label = label + '00'
        profiles = pd.concat([profiles, pd.DataFrame([[label, r, t]],
                                                     columns=['profile label', 'radius', 'thickness'])])
    profiles.to_csv('full_structure/structure1_fullstruct.pro', index=False)


def smaller_profile(profile_label):
    profiles = pd.read_csv('full_structure/structure1_fullstruct.pro')
    current_idx = profiles.index[profiles['profile label'] == profile_label][0]
    if current_idx > 0:
        current_idx = current_idx - 1
    return profiles.iloc[current_idx]['profile label']


def larger_profile(profile_label):
    profiles = pd.read_csv('full_structure/structure1_fullstruct.pro')
    current_idx = profiles.index[profiles['profile label'] == profile_label][0]
    if current_idx < len(profiles) - 1:
        current_idx = current_idx + 1
    return profiles.iloc[current_idx]['profile label']


if __name__=='__main__':
    generate_profiles(0.001, 1, 20)
    # smaller_profile('P1.0')