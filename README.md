# Generative Adversarial Networks for improved sampling of molecular trajectories in molecular machine learning

## Setup

### 1. Install python 3.10
```shell
sudo apt install python3.10
```

### 2. Check the installation
```shell
python3.10 --version
```

### 3. Clone the repository
```shell
git clone https://github.com/panthibivek/Generative-Adversarial-Network-for-Improving-Sampling-of-Molecular-Trajectories
```
### 4. CD into the repository
```shell
cd GAN_pkg/
```
### 5. Install all the requirements
```shell
pip install -r requirements.txt
```

## The repository tree structure
.
├── formatData.py
├── gan.py
├── generatexyz.py
├── reformatToXyz.py
├── coulombToTraj.py
├── train.py
└── utils.py
├── benchmark.ipynb
├── energy_benchmarking.ipynb
├── genAdvTrain.ipynb
├── generateSamples.ipynb
├── optimzation.ipynb
├── regression.ipynb
├── runs
│   ├── ModelForCoulombToXyz
│   └── train
├── config_files_orca
│   ├── calculated_energies.txt
│   ├── extract_excited_energy.sh
│   ├── gan_lower_coulomb_mtx_array.txt
│   ├── molecule.log
│   ├── molecule.xyz.inp
│   ├── runOrca.py
│   └── run.sh
├── data
│   ├── AllMolecules
│   ├── energies.txt
│   ├── example_visualization
│   ├── example_xyz_file_from_gan.xyz
│   ├── exp
│   ├── flattenedXyz.txt
│   ├── lower_coulomb_mtx_array.txt
│   ├── MD
│   ├── MoleculesMappedFromSampleSpace
│   ├── newMappedMolecules
│   ├── Traj1
│   ├── Traj2
│   └── traj.xyz
├── README.md
├── requirements.txt

List of class used in the repository:

## GenAdvNetwork:
    \t This class is used to compile and train the GAN model.

    \t ### Methods:
        #### __init__:
            \t \t Class constructor
        #### generate_generator:
            \t \t Function that defines the architecture of the generator
        #### generate_discriminator:
            \t \t Function that defines the architecture of the discriminator
        #### compile:
            \t \t Function that initializes the loss functions used for the generator and the discriminator
        #### generate_trajectories:
            \t \t Function that uses the generator to generate new trajectories
        #### train_disc_gen:        
            \t \t Function that trains the generator and the discriminator
        #### train_step:
            \t \t Function that trains the GAN for each batch of data