#!/bin/bash

## Activate virtual environment
source .venv/bin/activate

## Update Environments
uv self update
uv lock --upgrade
uv sync --all-extras --all-groups

## Code style
uv run isort .
uv run mypy .
uv run pyre
uv run ruff format
uv run ruff check --fix

## Update docs 
quarto render

## Check test 
uv run pytest

