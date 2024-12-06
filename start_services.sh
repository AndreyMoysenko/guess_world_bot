#!/bin/bash

celery -A src.api.tasks.celery worker --loglevel=info --pool=solo &
python src/api/app.py
tail -f /dev/null