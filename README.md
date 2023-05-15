# Generative Adversarial Networks for improved sampling of molecular trajectories in molecular machine learning

Note: The code requires a Linux based operating system. <br>

## Setup

### 1. Install python 3.10
```shell
sudo apt install python3.10
```

### 2. Check the installation
```shell
python3.10 --version
```

### 3. Install virtual env
```shell
sudo apt-get install python3-venv
```

### 4. Create a directory for the environment
```shell
mkdir GAN_pkg
```

### 5. CD into the directory
```shell
cd GAN_pkg/
```

### 6. Create a new virtual environment
```shell
python3 -m venv myenv
```

### 7. Activate the virtual environment 
```shell
source myenv/bin/activate
```

### 8. Clone the repository
```shell
git clone https://github.com/panthibivek/Generative-Adversarial-Network-for-Improving-Sampling-of-Molecular-Trajectories.git
```

### 9. CD into the repository
```shell
cd Generative-Adversarial-Network-for-Improving-Sampling-of-Molecular-Trajectories/
```

### 10. Install all the requirements
```shell
pip install -r requirements.txt
```

### 11. Check if all the packages are installed 
```shell
pip freeze
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
&emsp; This class is used to compile and train the GAN model.<br>
<br>
&emsp; **Methods**:<br>
<br>
&emsp; &emsp; **\_\_init\_\_**:<br>
&emsp; &emsp; &emsp; &emsp; Class constructor<br>
&emsp; &emsp; **generate_generator**:<br>
&emsp; &emsp; &emsp; &emsp; Function that defines the architecture of the generator<br>
&emsp; &emsp; **generate_discriminator**:<br>
&emsp; &emsp; &emsp; &emsp; Function that defines the architecture of the discriminator<br>
&emsp; &emsp; **compile**:<br>
&emsp; &emsp; &emsp; &emsp; Function that initializes the loss functions used for the generator and the discriminator<br>
&emsp; &emsp; **generate_trajectories**:<br>
&emsp; &emsp; &emsp; &emsp; Function that uses the generator to generate new trajectories<br>
&emsp; &emsp; **train_disc_gen**:<br>
&emsp; &emsp; &emsp; &emsp; Function that trains the generator and the discriminator<br>
&emsp; &emsp; **train_step**:<br>
&emsp; &emsp; &emsp; &emsp; Function that trains the GAN for each batch of data<br>


### GenerateCsv:
&emsp; This class is used to seperate the molecules into individual XYZ files and generate Coulomb matrix for each molecules.<br>
<br>
&emsp; **Methods**:<br>
<br>
&emsp; &emsp; **\_\_init\_\_**:<br>
&emsp; &emsp; &emsp; &emsp; Class constructor<br>
&emsp; &emsp; **GenerateXYZFiles**:<br>
&emsp; &emsp; &emsp; &emsp; Function that seperates the molecules into individual XYZ files<br>
&emsp; &emsp; **CreateCoulombMtx**:<br>
&emsp; &emsp; &emsp; &emsp; Function that generates Coulomb matrix for each molecules<br>
&emsp; &emsp; **loadData**:<br>
&emsp; &emsp; &emsp; &emsp; Function that loads the Coulomb matrix vectors and respective energies into numpy arrays<br>


### Generatexyz:
&emsp; This class is used to generate large sample of molecules for KD Tree search algorithm<br>
<br>
&emsp; **Methods**:<br>
<br>
&emsp; &emsp; **\_\_init\_\_**:<br>
&emsp; &emsp; &emsp; &emsp; Class constructor<br>
&emsp; &emsp; **generate_samples**:<br>
&emsp; &emsp; &emsp; &emsp; Function that generates specified number of 3D molecular coordinates<br>
&emsp; &emsp; **generate_samples_batch**:<br>
&emsp; &emsp; &emsp; &emsp; Function that generates one batch size of 3D molecular coordinates<br>
&emsp; &emsp; **generate_coulomb_matrix**:<br>
&emsp; &emsp; &emsp; &emsp; Function that generates the Coulomb matrx for all molecules<br>
&emsp; &emsp; **format_xyz_samples**:<br>
&emsp; &emsp; &emsp; &emsp; Function that reformats the newly generated molecules to the XYZ file format<br>
&emsp; &emsp; **generate_gaussian_noise**:<br>
&emsp; &emsp; &emsp; &emsp; Function that generates Gaussian noise<br>
&emsp; &emsp; **sorting_by_coulomb_matrix**:<br>
&emsp; &emsp; &emsp; &emsp; Function that sorts all the molecules by the squared distance of each Coulomb matrix vector from origin in ascending order<br>


### MapCoulomb1dToXYZ:
&emsp; This class is used to construct a KD tree and map each Coulomb matrix to 3D molecular coordinates<br>
<br>
&emsp; **Methods**:<br>
<br>
&emsp; &emsp; **\_\_init\_\_**:<br>
&emsp; &emsp; &emsp; &emsp; Class constructor. This method also constructs the KD tree.<br>
&emsp; &emsp; **generateXYZ**:<br>
&emsp; &emsp; &emsp; &emsp; Function that takes the array of Coulomb matrces, finds the 3D molecular coordinates for each molecule and reformats them to XYZ file format<br>
&emsp; &emsp; **getxyz**:<br>
&emsp; &emsp; &emsp; &emsp; Function that returns the 3D molecular coordinates given an index<br>
&emsp; &emsp; **findSpecificXyzIndex**:<br>
&emsp; &emsp; &emsp; &emsp; Function that querys the index of the nearest 3D molecular coordinates for a Coulomb matrix<br> <br>


## The use of each classes are shown in the following Jupyter Nootbooks:

**genAdvTrain.ipynb**:<br>
&emsp; &emsp; Building the Generative Adversarial Network (GAN)<br>

**regression.ipynb**:<br>
&emsp; &emsp; Building the Baseline Experiment (KRR Model)<br>

**optimzation.ipynb**:<br>
&emsp; &emsp; Testing a Multi-Layer Perceptron to map 1D representation of a Coulomb Matrix to XYZ coordinates<br>

**generateSamples.ipynb**:<br>
&emsp; &emsp; Generating and mapping 1D representation of a Coulomb Matrix to XYZ coordinates using KD Tree algorithm<br>

**energy_benchmarking.ipynb**:<br>
&emsp; &emsp; Visualizing the energies for Benzene Molecues generated by GAN<br>

**benchmark.ipynb**:<br>
&emsp; &emsp; Benchmarking the performance of KRR model with resampled training set<br> <br>


## Data Requirement for running the Jupyter Nootbooks:
Note: Please place all the data files in data directory

**Energy File**: /data/energies.txt <br>
**XYZ File**: /data/traj.xyz <br>

**Energy File**: /data/MD/traj1_energies.txt <br>
**XYZ File**: /data/MD/trajectory1.xyz <br>

**Energy File**: /data/MD/traj2_energies.txt <br>
**XYZ File**: /data/MD/trajectory2.xyz <br>
