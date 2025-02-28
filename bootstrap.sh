#!/bin/sh
export FLASK_APP=./index1.py
export PIPENV_IGNORE_VIRTUALENVS=1
pipenv run flask --debug