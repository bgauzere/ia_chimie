# Install poetry

`/usr/bin/pip3 install poetry  --user`
 
 
#  Install dependencies

 1. `git clone https://github.com/bgauzere/ia_chimie.git`
 2. `cd ia_pour_la_chimie`
 3. `python3 -m poetry install`
 4. `python3 -m poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
	
 5. Test	de la config:
	* `poetry run python`
	* `import xgboost`
	* `import torch`
	
 6. Installation du kernel pour jupyter
    * `poetry run python -m ipykernel install --name=ia_chimie --user`
	
# Lancement du notebook #

`python3 -m poetry run jupyter notebook &`
 
 Then you should be ok to work on pratical sessions.
