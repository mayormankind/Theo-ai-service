#!/bin/bash

# Start the FastAPI app with proper host and port binding for Render
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
