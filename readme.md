1. `uv sync --extra cu124`  
make sure CUDA is available (>12.4)
2. setup your .env file:  
```
// Example
DATABRICKS_HOST="https://**********.cloud.databricks.com"
DATABRICKS_TOKEN="dap**************"
DATABRICKS_EXPERIMENT="/Users/******@gmail.com/test"
```

3. `uv run --env-file=.env train.py`
