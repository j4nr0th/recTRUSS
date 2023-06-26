from sys import argv

import pandas as pd
from numpy import pi

if __name__ == "__main__":
    if len(argv) != 2:
        print("Need one argument only")
        exit(1)

    df = pd.read_csv(argv[1], sep=',')
    radii = df["radius"]
    thicknesses = df["thickness"]
    A = pi * (radii ** 2 - (radii - thicknesses) ** 2)
    I = pi / 4 * (radii ** 4 - (radii - thicknesses) ** 4)
    new_df = pd.DataFrame({"profile label": df["profile label"], "A": A, "I": I})
    new_df.to_csv(argv[1] + "_new.csv")