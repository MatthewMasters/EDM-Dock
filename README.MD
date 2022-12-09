# EDM-Dock

Code for our paper [**Deep Learning Model for Flexible and Efficient Protein-Ligand Docking**]()

## Installation

```
git clone https://github.com/MatthewMasters/EDM-Dock
cd EDM-Dock
conda env create -f environment.yaml -n edm-dock
conda activate edm-dock
python setup.py install
```

## Usage


### Dock your own molecules using the pre-trained model
#### Step 1: Prepare a dataset using the following format
```
dataset_path/
    sys1/
        protein.pdb
        ligand.sdf
        box.csv
    sys2/
        protein.pdb
        ligand.sdf
        box.csv
    ...
```
The `box.csv` file defines the binding site box and should have six comma-seperated values:
```
center_x, center_y, center_z, width_x, width_y, width_z
```
#### Step 2: Prepare the features using the following command
```
python scripts/prepare.py --dataset_path [dataset_path]
```
#### Step 3: Download DGSOL
Since DGSOL does not have an MIT license, it's code is included in a seperate repository (https://github.com/MatthewMasters/DGSOL.git).
Once you have downloaded DGSOL independently, update the path at the top of `edmdock/utils/dock.py` to reflect the path on your system.
Remember to rebuild the package by issuing the command `python setup.py install`.

#### Step 4: Run Docking
By default this will run the docking including the minimization process.
You can turn off minimization for much faster docking, however it may generate unrealistic molecular structures by editing the last line in `runs/paper_baseline/config.yml`.
```
python scripts/dock.py --run_path runs/paper_baseline --dataset_path [dataset_path]
```
The final docked poses are saved in the folder `runs/paper_baseline/results` as `[ID]_docked.pdb`.

### Train model using your own dataset

#### Step 1: Prepare a dataset using the format described above
#### Step 2: Prepare the features using the following command 
```
python scripts/prepare.py --dataset_path [dataset_path]
```
#### Step 3: Write a configuration file
An example can be found at `runs/paper_baseline/config.yml`
#### Step 4: Begin training with the following command
```
python scripts/train.py --config_path [config_path]
```

### Reference
```
Under Review
```


