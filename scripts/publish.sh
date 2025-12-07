#!/bin/bash

rm -rf dist
uv build
uv publish