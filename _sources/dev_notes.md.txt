## Project

### Project specifications

```mermaid
classDiagram
	class DataContainer{
    <<abstract>>
		+UUID uid
        +str external_uid
		+str label
        +set[LinguisticRealisation] linguistic_realisations

		+add_linguistic_realisation(LinguisticRealisation)
		+remove_linguistic_realisation()
	}

	class Concept {
	}

	class Relation{
		+Concept source_concept
		+Concept destination_concept
	}

	class MetaRelation{
	}

	class KnowledgeRepresentation{
		+set[Concept] concepts
		+set[Relation] relations
		+set[MetaRelation] metarelations
	}

	class LinguisticRealisation {
		+str label
		+Enrichment enrichment
		+set[Any] corpus_occurrences

    +get_docs() set[Spacy.doc]
	}

    class ConceptLR{
        +set[Spacy.span] corpus_occurrences

    +add_corpus_occurrences(set[Spacy.span])
    }

    class RelationLR{
        +set[tuple[Spacy.span, Spacy.span, Spacy.span]] corpus_occurrences

    +add_corpus_occurrences(set[tuple[Spacy.span, Spacy.span, Spacy.span]])
    }

    class MetaRelationLR{
        +set[tuple[Spacy.span, Spacy.span]] corpus_occurrences

    +add_corpus_occurrences(set[tuple[Spacy.span, Spacy.span]])
    }

	class Enrichment {
		+set[str] synonyms

    +add_synonyms(Set[str])
	}

    class CandidateTerm {
        +str label
        +set[Spacy.span] corpus_occurrences
        +Enrichment enrichment

    +add_corpus_occurrences(set[Spacy.span])
    }

	class Pipeline{
		<<abstract>>
		+list[PipelineComponent] pipeline_components
		+list[DataPreprocessing] preprocessing_components
        +Spacy.lang spacy_model
        +CorpusLoader corpus_loader
		+list[Spacy.doc] corpus
		+KnowledgeRepresentation kr
        +set[CandidateTerm] candidate_terms

        +add_pipeline_component(PipelineComponent)
        +remove_pipeline_component(PipelineComponent)
		+build()
		+run()

	}

    class CorpusLoader{
        <<abstract>>
        +str corpus_path

        -read_corpus() list[str]
    }

    class JsonCorpusLoader{
        +str json_field
    }

    class CsvCorpusLoader{
    }

	class PipelineComponent{
		<<abstract>>

		+check_resources()
		+optimise()
		-compute_metrics()
		+get_performance_report() Dict[str, Any]
		+run(Pipeline)
	}

	class DataPreprocessing{
		<<abstract>>
		+dict[str, Any] parameters
		+dict[str, Any] options

		+run(Pipeline)
	}
	class ConceptRelationExtraction
	class CandidateTermExtraction
	class MetarelationExtraction
	class CandidateTermEnrichment
	class AxiomExtraction

    class KnowledgeSource{
		<<abstract>>
        +dict[str, Any] parameters

        +check_resources()
        -check_parameters()
		+match_external_concepts(term: CandidateTerm) Set[str]
        +enrich_term(term: CandidateTerm)
	}

    ConceptRelationExtraction --> KnowledgeSource : uses
    CandidateTermEnrichment --> KnowledgeSource : uses

	PipelineComponent <|--ConceptRelationExtraction
	PipelineComponent <|--CandidateTermExtraction
	PipelineComponent <|--MetarelationExtraction
	PipelineComponent <|--CandidateTermEnrichment
	PipelineComponent <|--AxiomExtraction

	Pipeline "1" *-- "0..n" DataPreprocessing
	Pipeline "1" *-- "3..n" PipelineComponent
    Pipeline "1" o-- "0..n" CandidateTerm

	Pipeline "1" *-- "1" KnowledgeRepresentation
	Pipeline "1" o-- "0..1" CorpusLoader

	CorpusLoader <|-- JsonCorpusLoader
	CorpusLoader <|-- CsvCorpusLoader

	KnowledgeRepresentation "1" *-- "0..n" Relation
	KnowledgeRepresentation "1" *-- "0..n" Concept
	KnowledgeRepresentation "1" *-- "0..n" MetaRelation
	DataContainer <|-- Concept
	DataContainer <|-- Relation
	Relation <|-- MetaRelation
	DataContainer "1" o-- "0..n" LinguisticRealisation
	LinguisticRealisation "1" o-- "0..1" Enrichment
	CandidateTerm "1" o-- "0..1" Enrichment
	Relation "1" o-- "2" Concept
    LinguisticRealisation <|-- ConceptLR
    LinguisticRealisation <|-- RelationLR
    LinguisticRealisation <|-- MetaRelationLR
```

### Project structure

Here is how the architecture is made:

- The folder `src/` contains all of the functional code.

  - The `src/commons/` folder contains all the tools used by the different modules.
  - The `src/data_container/` folder contains all the data structure needed for the framework.
  - The `src/pipeline/` folder contains algorithms and pipeline structure to build and run an ontology learning model.

- The folder `demonstrators/` contains demonstrators for specific pipeline components or different ontology learning processes.

About the code structure:

- Generic data structure are identified with `schema.py` at the end of the filename.
- Each new class must be tested and the test file is identified with `test`at the beginning of the filename.

## Code

### Coding style

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

- [Numpy docstrings style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

### Test

Every development must be tested.
The tests are launched with the `pytest` command.

As we want to be an open source library, we must have a test coverage percentage close to 100%.
The following command should be used to evaluate:

```Bash
pytest --cov-report term-missing --cov=. test
```

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

Update the code after an amend commit

```
git checkout {branch_to_be_merged}
git fetch
git reset --hard {origin}/{branch_to_be_merged}
```

### Virtual environment

Setting up the virtual environment:

- go to the project root directory: `ontology-learning/`
- create the virtual environment by running `virtualenv -p python3.10 {env/path}` (virtualenv needs to be installed) or `python3 -m venv {env/path}`. As a result, you should have a new folder at your project root.
- activate the virtual environment by running
  - `source {env/path}/bin/activate` on Linux
  - `{env/path}\Scripts\activate` on Windows
- check that the environment is properly installed by running the test from the project root directory with the command line `pytest test`.

Setting up the workspace :

- add the `src/` folder to the python paths by adding the full path `{path/to/the/project/}ontology-learning/src` to the file:
  - on Linux add `{path/to/the/project/}ontology-learning/src` to the file `ontology-learning/venv/lib/python3.10/site-packages/_virtualenv.pth`
  - on Windows add `{C:\path\to\the\project\}ontology-learning\src` to the file `{path/to/the/project/}ontology-learning\venv\Lib\site-packages\_virtualenv.pth`
- instead, the following command can also be run each time the project is used : `export PYTHONPATH="${PYTHONPATH}:{path/to/the/project/}ontology-learning/src"`

Setting up project dependencies :

- install the project dependencies by running `pip install -r requirements.txt` (from within the virtual environment)
- update the requirements after new downloads by running `pip freeze > requirements.txt`

### Documentation

Generate documentation :

- move to gh-pages `git checkout gh-pages`
- install sphinx via pip `pip install sphinx`
- move to docs folder `cd docs`
- initialize docs `sphinx-quickstart`
- go back to the root folder `cd ..`
- generate sphinx markdown `sphinx-apidocs -o docs olaf/`
- and then generate html pages `cd docs && make html`

To host the documentation with with github-pages :

- create a branch named `gh-pages`
- enable github pages and set the working branch as `gh-pages`
- create an acces-token
- add an environment secret named `TOKEN` with the generated acces token as value
- add an environment variable called `USER_MAIL` with your email address as value
- add an environment variable called `USER_NAME` with your github username as value