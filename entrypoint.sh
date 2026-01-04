
#!/bin/bash
set -e

# Activate virtual environment (just in case)
source .venv/bin/activate

# Logic to handle the specific commands requested
case "$1" in
    "train")
        echo "Running Training..."
        python scripts/train.py
        ;;
    "inference")
        echo "Running Inference..."
        python scripts/batch_inference.py
        ;;
    *)
        # If the user types something else (like bash), run that
        exec "$@"
        ;;
esac
