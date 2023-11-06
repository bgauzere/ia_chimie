# Install poetry

3 solutions. Try from the first one until the third if it doesn't work.

 1. `curl -sSL https://install.python-poetry.org | python3 -`
 2. `pipx install poetry`
 3. `pip install poetry --user`
 
 
#  Install dependencies

 1. `git clone ...`
 2. `cd ia_pour_la_chimie`
 3. `poetry install`
 4. `poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
 5. Test	de la config:
	* `poetry run python`
	* `import xgboost`
	* `import torch`
 6. Installation du kernel pour jupyter
    * `poetry run python -m ipykernel install --name=ia_chimie --user`
	
 Then you should be ok to work on pratical sessions.
