# Logbook

## 20/07/2023

### custom spaCy doc attribute for selected token sequences (MR !55)

The content of the custom spaCy doc attribute for selected token sequences has to be spans. It is expected by the pipeline components using it. Otherwise, we will have to do a type check within the components.

We can have any different custom spaCy doc attribute for selected token sequences on the same corpus documents. Though only one of those custom attributes is usable for each component.

### About options (MR !55)

We agree that options are values used by the optimise function.

Every option can be a parameter. The fact that a parameter is defined in the options dict and not in the parameter one will be used by the optimise method to know that this "parameter" is optimisable. The optimise method should receive as input an evaluation dataset and the values (numerical range or set of values) to test in a grid search manner for each parameter (option) we wish to optimise on.
In the end, placing a parameter in the parameter dict or the option one comes down to deciding as a developer which parameter we let the user tune. Of course, not all parameters can be tunable options, but we would assume that any option can be a param.

rationals:
- Having options containing the range of values to be optimized, e.g., `parameters = {"threshold": 0.7}` and `options = {"threshold": [0.2,0.9,0.1]}` would force the user to predefine the search space for optimisation before creating the component (though we can always change the options manually after creation). It sounds quite complicated.

Pipeline optimisation user journey:

- The optimise methods should update the component attributes and return the best metric value. 
- The components will be run by the pipeline run.
- The user journey should be:

  1. define your components
  2. Build the pipeline, i.e., assign components to pipelines
  3. optimise the components you want
  4. run the pipeline
  
## 17/07/2023

### Pipeline schema

From an OLAF user point of view, it is better to instantiate a Pipeline than extend the Pipeline to create a custom one.
That's why we removed the Pipeline class as an abstract one.

### Pipeline Component schema

We add the pipeline as parameter for the run method in pipeline component so that each algorithm has access to all information available.
However, as the pipeline has pipeline components, this creates circular import and thus errors. 
To solve that, we declare the pipeline as Any type in the run method of pipeline component even if we know that it has a Pipeline type to solve the problem. We will have to check that this not cause other problems.

## 28/06/2023

User typical OLAF usage:

1. Import pipeline components
2. Instantiate pipeline components
3. Instantiate pipeline
4. Run pipeline

### Pipeline components

Pipeline components need access to the pipeline instance to directly update pipeline attributes, e.g., KR, candidate terms ...
The solution suggested is to pass the pipeline as argument to the pipeline component run method: `pipeline_component.run(pipeline)`.

### Data preprocessing

- The tokeniser is loaded with the spaCy model
- The preprocessing pipeline components enrich the existing documents
- Each preprocessing step is an individual component, e.g., lemmatisation,
- Data preprocessing process:
  - If there is no corpus provided use the corpus loader with the spaCy model
  - If there are some preprocessing components, then apply them to the corpus

### Linguistic realisations

(Discussion, no definitive decisions)

We could simply the structure by keeping only the LinguisticRealisation class and have optional source, relation, and destination attributes.
Wether it is a concept LR, relation LR or metarelation LR will be inferred by the pipeline component based on the implement attributes.
