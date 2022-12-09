# EDM-Dock

Code for our paper [**Deep Learning Model for Flexible and Efficient Protein-Ligand Docking**]()

## Installation

```
git clone https://github.com/MatthewMasters/EDM-Dock
conda env create -f environment.yml -n edm-dock
conda activate edm-dock
cd EDM-Dock
python setup.py install
```

## Usage


### Dock your own molecules using the pre-trained model
#### Step 1: Prepare a dataset using the following format
```
dataset_path/
    sys1/
        protein.pdb
        ligand.mol2
        box.csv
    sys2/
        protein.pdb
        ligand.mol2
        box.csv
    ...
```
The `box.csv` file defines the binding site box and should have six comma-seperated values:
```
center_x, center_y, center_z, width_x, width_y, width_z
```
#### Step 3: Prepare the features using the following command
```
python scripts/prepare.py --dataset_path [dataset_path]
```
#### Step 4: Edit configuration files



### Train model using your own dataset

### Reference
```
Under Review
```


