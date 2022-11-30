# Ontology-learning

## project structure

### Folder structure

We follow the layer cake architecture.

- The folder `src/` structure is inspired and follows the ontology learning layer cake presented in <https://doi.org/10.1093/database/bay101>.
  - The `src/config/` folder contains all the configuration management files.
  - Each of the other folders contain a main class in the file `src/{layer cake name}/{layer cake name}_service.py`. This class is acting as a "listing"
  
Code structure:

- All main classes (i.e., the ones defined in the `src/{layer cake name}/{layer cake name}_service.py`) are initialized with a corpus object of format: `List[Spacy.Doc]`
- Only the `Datapreprocessing` class (`src/data_preprocessing/data_preprocessing_service.py`) takes the corpus with format: `List[str]`

### Configuration management

The project is set up to have only one global configuration file `config.cfg` located in the `src/config/` folder.
We rely on the python library [`confection`](https://github.com/explosion/confection)

We only push an example version of a config file
It is the library user job to copy her/his own configuration file into the `src/config/` folder.

The configuration file follows the below structure:

- One section per ontology learning layer cake (i.e., data_preprocessing, term_extraction, etc.)
- One subsection for each sub-components specific configuration (e.g., token_selector.select_on_pos with a key-value pair pos_to_select = '["NOUN", "VERB"]')

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

```Bash
git pull 
git checkout -b {new_branch}
```

Submit new developments

```Bash
git add {files}
git commit -m "{comment}"
git push {origin} {new_branch}
```

Amend commit and push after a review

```Bash
git checkout {branch_to_be_merged}
git add {modified_files}
git commit --amend
git push --force-with-lease {origin} {branch_to_be_merged}
```

## Virtual environment

Setting up the virtual environment:

- change directory to the project root directory: `ontology-learning/`
- run `virtualenv -p python3.8 venv`. (Virtualenv needs to be installed)
- add the `src/` folder to the python paths by adding the full path `{path/to/the/project/}ontology-learning/src` to the file:
  - On Linux add `{path/to/the/project/}ontology-learning/src` to the file `ontology-learning/venv/lib/python3.8/site-packages/_virtualenv.pth`
  - On Windows add `{C:\path\to\the\project\}ontology-learning\src` to the file `{path/to/the/project/}ontology-learning\venv\Lib\site-packages\_virtualenv.pth`
- Install the project dependencies: `pip install -r requirements.txt`

Update requirements.txt: `pip freeze > requirements.txt`

Run virtual environment:

- On Linux: `source {path/to/the/project/}venv/bin/activate`
- On Windows: `{C:\path\to\the\project\}venv\Scripts\activate`

## Spacy language processing pipelines (Deprecated)

To create and save a Language processing pipeline you need to:

- Create your pipeline (and test it)
- Save it to disk in the `SPACY_PIPELINE_PATH` directory using the [spacy Language `to_disk()` method](https://spacy.io/api/language#to_disk): `your_spacy_model .to_disk(os.path.join(SPACY_PIPELINE_PATH, "your_model_name"))`
- It will create a folder `SPACY_PIPELINE_PATH/your_model_name/` with all the details to reconstruct your pipeline.
- You can edit the `SPACY_PIPELINE_PATH/your_model_name/meta.json` file to add your personal information and associate the pipeline with you as an author.

You can then load it anywhere in the project with: `your_spacy_model = spacy.load(os.path.join(SPACY_PIPELINE_PATH, "your_model_name"))`