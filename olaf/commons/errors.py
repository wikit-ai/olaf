class ResourcesCheckFailError(Exception):
    """Exception raised when a resource is missing to run a component of the pipeline."""

    def __init__(self) -> None:
        message = (
            f"External resources check failed. Some might not be accessible or wrong."
        )
        super().__init__(message)


class MissingEnvironmentVariable(Exception):
    """Exception raised when an environment variable is missing."""

    def __init__(self, component_name: str, env_var_name: str) -> None:
        message = f"""External resources check failed for component {component_name}.
                    Missing environment variable: {env_var_name}"""
        super().__init__(message)

class ParameterError(Exception):
    """Exception raised when a required parameter is missing for a pipeline component to function."""

    def __init__(self, component_name: str, param_name: str, error_type: str) -> None:
        """Initialise a parameter error.

        Parameters
        ----------
        component_name: str
            The name of the pipeline component the exception comes from.
        param_name: str
            The name of the parameter causing the exception.
        error_type: str
            The kind of error associated with the exception.

        """
        message = f"""A parameter error occurred while initialising pipeline component {component_name} due to parameter {param_name}.
                        Parameter Error type: {error_type}
                """
        super().__init__(message)


class OptionError(Exception):
    """Exception raised when a required option is missing for a pipeline component to function."""

    def __init__(self, component_name: str, option_name: str, error_type: str) -> None:
        """Initialise an option error.

        Parameters
        ----------
        component_name: str
            The name of the pipeline component the exception comes from.
        option_name: str
            The name of the option causing the exception.
        error_type: str
            The kind of error associated with the exception.

        """
        message = f"""A option error occurred while initializing pipeline component {component_name} due to option {option_name}.
                        Option Error type: {error_type}
                """
        super().__init__(message)


class EmptyCorpusError(Exception):
    """Exception raised when the text corpus represented as spacy documents is empty."""

    def __init__(self) -> None:
        message = f"Corpus is empty. No documents were given or the spacy pipe process failed."
        super().__init__(message)


class PipelineCorpusInitialisationError(Exception):
    """Exception raised when a pipeline is initialised without corpus nor corpus loader."""

    def __init__(self) -> None:
        message = (
            f"Pipeline can not be initialised without a corpus or a corpus loader."
        )
        super().__init__(message)


class FileOrDirectoryNotFoundError(Exception):
    """Exception raised when the corpus path is not a directory or a file."""

    def __init__(self, path: str) -> None:
        message = f"The corpus path {path} is not a valid directory or file."
        super().__init__(message)


class NotCallableError(Exception):
    """Exception raised when the argument passed as a function is not callable."""

    def __init__(self, function_name: str) -> None:
        message = f"The function {function_name} is not callable."
        super().__init__(message)
