#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:23:25 2020

@author: ljia
"""
import os
import re


cur_path = os.path.dirname(os.path.abspath(__file__))


def get_job_script(arg):
	script = r"""
#!/bin/bash

#SBATCH --exclusive
#SBATCH --job-name="st.""" + arg + r""".bp"
#SBATCH --partition=court
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output="outputs/output_edit_costs.real_data.nums_sols.ratios.bipartite.""" + arg + """.txt"
#SBATCH --error="errors/error_edit_costs.real_data.nums_sols.ratios.bipartite.""" + arg + """.txt"
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000

srun hostname
cd """ + cur_path + r"""
echo Working directory : $PWD
echo Local work dir : $LOCAL_WORK_DIR
python3 edit_costs.real_data.nums_sols.ratios.bipartite.py """ + arg
	script = script.strip()
	script = re.sub('\n\t+', '\n', script)
	script = re.sub('\n +', '\n', script)

	return script

if __name__ == '__main__':

	os.makedirs('outputs/', exist_ok=True)
	os.makedirs('errors/', exist_ok=True)

	ds_list = ['Acyclic', 'Alkane_unlabeled', 'MAO_lite', 'Monoterpenoides', 'MUTAG']
	for ds_name in [ds_list[i] for i in [0, 1, 2, 3, 4]]:
		job_script = get_job_script(ds_name)
		command = 'sbatch <<EOF\n' + job_script + '\nEOF'
# 		print(command)
		os.system(command)
# 		os.popen(command)
# 		output = stream.readlines()
