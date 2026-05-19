#!/bin/bash
gunicorn basic_chatbot1:app --bind 0.0.0.0:$PORT
