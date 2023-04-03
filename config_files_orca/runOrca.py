
""" @author: Bivek
"""
import os
import subprocess
import numpy as np

if __name__=="__main__":
    path_to_molecules = os.path.abspath(os.path.dirname(__file__)) + "/MoleculesMappedFromSampleSpace/"
    input_filename = os.path.abspath(os.path.dirname(__file__)) + "/molecule.xyz.inp"
    sh_filename_for_run = os.path.abspath(os.path.dirname(__file__)) + "/run.sh"
    sh_filename_for_extracting_energy = os.path.abspath(os.path.dirname(__file__)) + "/extract_excited_energy.sh"
    calculated_energies_filename = os.path.abspath(os.path.dirname(__file__)) + "/calculated_energies.txt"
    all_calculated_energies = []

    for path in sorted(os.listdir(path_to_molecules)):
        current_molecule = path_to_molecules + str(path)
        print(current_molecule)
        with open(input_filename, 'r+') as xyz_file:
            all_lines = xyz_file.readlines()
            all_lines[-1] = "* xyzfile 0 1 {}\n".format(current_molecule)
            xyz_file.seek(0)
            xyz_file.writelines(all_lines)
            xyz_file.truncate()
        process = subprocess.run(['sh', sh_filename_for_run])
        result = subprocess.run(['sh', sh_filename_for_extracting_energy], capture_output=True)
        energy_str = str(result.stdout.decode())
        # energy = int(energy_str[0:-2])
        all_calculated_energies.append(energy_str)
    

    calculated_energies_file = open(calculated_energies_filename, "w")
    energy_string = ''.join(str(energy) for energy in all_calculated_energies)
    calculated_energies_file.write(energy_string)
    calculated_energies_file.close()


