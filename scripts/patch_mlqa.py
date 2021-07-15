"""
Copyright (c) Facebook, Inc. and its affiliates.

The squad eval scripts checks the version on its dataset and errors
if it is incorrect. This script:
- patches the version number on the MLQA
- If the version is 2.0, then also patch the questions to set is_impossible=False
"""

from pathlib import Path

import typer
from pedroai.io import read_json, write_json
from rich.console import Console

cli = typer.Typer()
console = Console()


@cli.command()
def v1(in_file: Path, out_file: Path):
    console.log(f"V1.1 Patching [cyan]{in_file}[/cyan] to [cyan]{out_file}[/cyan]")
    dataset = read_json(in_file)
    dataset["version"] = "1.1"
    write_json(out_file, dataset)


@cli.command()
def v2(in_file: Path, out_file: Path):
    console.log(f"V2.0 Patching [cyan]{in_file}[/cyan] to [cyan]{out_file}[/cyan]")
    dataset = read_json(in_file)
    dataset["version"] = "2.0"
    for page in dataset["data"]:
        for paragraph in page["paragraphs"]:
            for question in paragraph["qas"]:
                question["is_impossible"] = False

    write_json(out_file, dataset)


if __name__ == "__main__":
    cli()
