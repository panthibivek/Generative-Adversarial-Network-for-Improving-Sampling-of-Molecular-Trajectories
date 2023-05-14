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
. <br>
├── formatData.py <br>
├── gan.py <br>
├── generatexyz.py <br>
├── reformatToXyz.py<br>
├── coulombToTraj.py<br>
├── train.py<br>
└── utils.py<br>
├── benchmark.ipynb<br>
├── energy_benchmarking.ipynb<br>
├── genAdvTrain.ipynb<br>
├── generateSamples.ipynb<br>
├── optimzation.ipynb<br>
├── regression.ipynb<br>
├── runs<br>
│   ├── ModelForCoulombToXyz<br>
│   └── train<br>
├── config_files_orca<br>
│   ├── calculated_energies.txt<br>
│   ├── extract_excited_energy.sh<br>
│   ├── gan_lower_coulomb_mtx_array.txt<br>
│   ├── molecule.log<br>
│   ├── molecule.xyz.inp<br>
│   ├── runOrca.py<br>
│   └── run.sh<br>
├── data<br>
│   ├── AllMolecules<br>
│   ├── energies.txt<br>
│   ├── example_visualization<br>
│   ├── example_xyz_file_from_gan.xyz<br>
│   ├── exp<br>
│   ├── flattenedXyz.txt<br>
│   ├── lower_coulomb_mtx_array.txt<br>
│   ├── MD<br>
│   ├── MoleculesMappedFromSampleSpace<br>
│   ├── newMappedMolecules<br>
│   ├── Traj1<br>
│   ├── Traj2<br>
│   └── traj.xyz<br>
├── README.md<br>
├── requirements.txt<br>

## List of class used in the repository:

### GenAdvNetwork:
&nbsp; This class is used to compile and train the GAN model.

&nbsp; Methods:
&nbsp; &nbsp; __init__:
&nbsp; &nbsp; &nbsp; Class constructor
&nbsp; &nbsp; generate_generator:
&nbsp; &nbsp; &nbsp; Function that defines the architecture of the generator
&nbsp; &nbsp; generate_discriminator:
&nbsp; &nbsp; &nbsp; Function that defines the architecture of the discriminator
&nbsp; &nbsp; compile:
&nbsp; &nbsp; &nbsp; Function that initializes the loss functions used for the generator and the discriminator
&nbsp; &nbsp; generate_trajectories:
&nbsp; &nbsp; &nbsp; Function that uses the generator to generate new trajectories
&nbsp; &nbsp; train_disc_gen:        
&nbsp; &nbsp; &nbsp; Function that trains the generator and the discriminator
&nbsp; &nbsp; train_step:
&nbsp; &nbsp; &nbsp; Function that trains the GAN for each batch of data