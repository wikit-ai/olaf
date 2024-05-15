# Ontology-learning

Since the beginning of the century, research on ontology learning has gained popularity. Automatically extracting and structuring knowledge relevant to a domain of interest from unstructured data is a major scientific challenge. We propose a new approach with modular ontology learning framework considering tasks from data pre-processing to axiom extraction. Whereas previous contributions considered ontology learning systems as tools to help the domain expert, we developed the proposed framework with full automation in mind.

Resources:

- [Poster](./docs/Poster_OLAF_2023.pdf)
- Our research paper has been published at [KES 2023](http://kes2023.kesinternational.org/).

## Installation

For usage :

```
pip install git+https://github.com/wikit-ai/olaf

```

For contribution :

```
git clone https://github.com/wikit-ai/olaf.git
cd olaf
python3 -m venv ./venv
source venv/bin/activate
pip install .
```

For demonstration : 

`olaf list` to display all pipeline demonstrations
`olaf show pipeline_demo_name` to display all pipeline components
`olaf run  all` to run all available pipeline demonstations
`olaf run  pipeline_demo_name` to run the specified pipeline demonstration

## Quick-start

A demo on how the library can be used is available in `demontrators/demo_test.ipynb`.

One example of OLAF usage for LLM components evaluation is also available here : [https://github.com/wikit-ai/olaf-llm-eswc2024](https://github.com/wikit-ai/olaf-llm-eswc2024).

## How to contribute

When an algorithm is missing you can contribute by adding it. Please refer to the [developer note](./docs/dev_notes.md) in the documentation for more detailed information.

## Citing us

> Marion Schaeffer, Matthias Sesboüé, Jean-Philippe Kotowicz, Nicolas Delestre, Cecilia Zanni-Merk,
> OLAF: An Ontology Learning Applied Framework,
> Procedia Computer Science,
> Volume 225,
> 2023,
> Pages 2106-2115,
> ISSN 1877-0509,
> https://doi.org/10.1016/j.procs.2023.10.201.
> (https://www.sciencedirect.com/science/article/pii/S1877050923013595)

## License

This project is licensed under the Apache-2.0 License.
