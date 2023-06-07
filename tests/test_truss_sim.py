import pathlib
import os
from main import main
os.chdir(pathlib.Path.cwd().parent)

cwd = pathlib.Path.cwd()
main(file_loc=str(cwd/'tests'/'structure1'), tot_dt_mass=0, drive_train_count=0, optimizing=False)