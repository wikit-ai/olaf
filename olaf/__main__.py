#:/usr/bin/python
import argparse
import importlib.util
import logging
import os
import re
import sys

from olaf.commons.logging_config import logger


def list_pipeline_names(module_name):
    # Import the module
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        logger.error("Module not found")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the directory containing the module
    module_dir = os.path.dirname(module.__file__)

    # List files in the module directory
    pipelines = [
        filename[:-3]
        for filename in os.listdir(module_dir)
        if (
            filename.endswith(".py")
            and not re.match(r"(__\w+__|_\w+)\.py", filename)
            and filename[:-3] != "runner"
        )
    ]
    return pipelines


def run_pipeline(pipeline_name):
    print(f"Running pipeline: {pipeline_name} \n")

    try:
        module = importlib.import_module(f"olaf.scripts.{pipeline_name}")
    except ModuleNotFoundError:
        logger.error("Unknown pipeline name.")
        list_pipelines()
        sys.exit(1)
    getattr(module, "PipelineRunner")().run(pipeline_name)


def list_pipelines():
    print("\nListing existing pipelines...")
    for pipeline in list_pipeline_names("olaf.scripts"):
        print(f"\t {pipeline}")


def show_pipeline(pipeline_name):
    print(f"\nShowing pipeline: {pipeline_name}")
    try:
        print(f"olaf.scripts.{pipeline_name}")
        module = importlib.import_module(f"olaf.scripts.{pipeline_name}")
    except ModuleNotFoundError:
        logger.error("Unknown pipeline name.")
        list_pipelines()
        sys.exit(1)

    getattr(module, "PipelineRunner")().describe()


def main():
    parser = argparse.ArgumentParser(
        description="Shortcuts for pipeline demonstration."
    )

    # Adding subparsers for different pipelines
    subparsers = parser.add_subparsers(dest="command", help="Available pipelines")

    # Subparser for the "run" pipeline
    run_parser = subparsers.add_parser("run", help="Run a pipeline.")
    run_parser.add_argument("pipeline", help="The pipeline to run.")

    # Subparser for the "list" pipeline
    subparsers.add_parser("list", help="List pipelines")

    # Subparser for the "show" pipeline
    show_parser = subparsers.add_parser("show", help="Describe a pipeline.")
    show_parser.add_argument("pipeline", help="The pipeline to show.")

    # Parse the arguments
    args = parser.parse_args()

    # Config logger to display only errors logging
    logger.setLevel(logging.ERROR)

    # Execute the appropriate function based on the pipeline
    if args.command == "run":
        if args.pipeline == "all":
            for pipeline in list_pipeline_names("olaf.scripts"):
                run_pipeline(pipeline)
        else:
            run_pipeline(args.pipeline)
    elif args.command == "list":
        list_pipelines()
    elif args.command == "show":
        show_pipeline(args.pipeline)


if __name__ == "__main__":
    main()
