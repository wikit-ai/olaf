# Ontology-learning



## Methodology

### Coding style
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

- [Numpy docstrings style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

### Git best practices

- Main branch must always be functional
- When developing a new feature, a new branch named "feat/{feature_name}" must be created
- When fixing an issue, a new branch named "fix/{feature_name}" must be created and linked to the issue
- Create a merge/pull request to submit the code

Start new developments
```
git pull 
git checkout -b {new_branch}
```
Submit new developments
```
git add {files}
git commit -m "{comment}"
git push {origin} {new_branch}
```
Amend commit and push after a review
```
git checkout {branch_to_be_merged}
git add {modified_files}
git commit --amend
git push --force-with-lease {origin} {branch_to_be_merged}
```
## Virtual environment

Setting up the virtual environment:

- change directory to the project root directory: `ontology-learning/`
- run `virtualenv -p python3.8 venv`. (Virtualenv needs to be installed)
- add the `src/` folder to the python paths by adding the full path `{path/to/the/project/}ontology-learning/src` to the file 
  - On Linux add `{path/to/the/project/}ontology-learning/src` to the file `ontology-learning/venv/lib/python3.8/site-packages/_virtualenv.pth`
  - On Windows add `{C:\path\to\the\project\}ontology-learning\src` to the file `C:\Users\msesboue\Documents\mindWork\ontology-learning\venv\Lib\site-packages\_virtualenv.pth`
- Install the project dependencies: `pip install -r requirements.txt`

Update requirements.txt: `pip freeze > requirements.txt`

Run virtual environment: 

- On Linux: `source {path/to/the/project/}venv/bin/activate`
- On Windows: `{C:\path\to\the\project\}venv\Scripts\activate`