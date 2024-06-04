#:/usr/bin/python
import os
import re
import sys
import argparse
import importlib.util


def list_pipeline_names(module_name):
    # Import the module
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print("Module not found")
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the directory containing the module
    module_dir = os.path.dirname(module.__file__)

    # List files in the module directory
    pipelines = []
    for filename in os.listdir(module_dir):
        # Check if the file is a Python module and doesn't match the pattern
        if (
            filename.endswith(".py")
            and not re.match(r"(__\w+__|_\w+)\.py", filename)
            and filename[:-3] != "runner"
        ):
            module_name = filename[:-3]  # Remove the ".py" extension
        pipelines.append(module_name)

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


def run_pipeline(args):
    print(f"Running pipeline: {args.pipeline} \n")

    try:
        module = importlib.import_module(f"olaf.scripts.{args.pipeline}")
    except ModuleNotFoundError:
        print("Invalid pipeline")
        sys.exit(1)
    getattr(module, "PipelineRunner")().run(args.pipeline)


def list_pipeline():
    print("Listing pipelines...")
    for pipeline in list_pipeline_names("olaf.scripts"):
        print("\t", pipeline)


def show_pipeline(args):
    print("Showing pipeline:", args.pipeline)
    try:
        print(f"olaf.scripts.{args.pipeline}")
        module = importlib.import_module(f"olaf.scripts.{args.pipeline}")
    except ModuleNotFoundError:
        print("Invalid pipeline")
        sys.exit(1)

    getattr(module, "PipelineRunner")().describe()


def main():
    parser = argparse.ArgumentParser(description="Your script description here")

    # Adding subparsers for different pipelines
    subparsers = parser.add_subparsers(dest="command", help="Available pipelines")

    # Subparser for the "run" pipeline
    run_parser = subparsers.add_parser("run", help="Run a pipeline")
    run_parser.add_argument("pipeline", help="The pipeline to run")

    # Subparser for the "list" pipeline
    subparsers.add_parser("list", help="List pipelines")

    # Subparser for the "show" pipeline
    show_parser = subparsers.add_parser("show", help="describe a pipeline")
    show_parser.add_argument("pipeline", help="The pipeline to show")

    # Parse the arguments
    args = parser.parse_args()

    # Execute the appropriate function based on the pipeline
    if args.command == "run":
        if args.pipeline == "all":
            for pipeline in list_pipeline_names("olaf.scripts"):
                run_pipeline(pipeline)
        else:
            run_pipeline(args)
    elif args.command == "list":
        list_pipeline()
    elif args.command == "show":
        if args.pipeline == "all":
            for pipeline in list_pipeline_names("olaf.scripts"):
                show_pipeline(pipeline)
        else:
            show_pipeline(args)


if __name__ == "__main__":
    main()
