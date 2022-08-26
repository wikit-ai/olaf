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

```

