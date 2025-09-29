import os
print("HOST:", os.getenv("DATABRICKS_HOST"))
print("TOKEN set?:", bool(os.getenv("DATABRICKS_TOKEN")))
print("EXP:", os.getenv("DATABRICKS_EXPERIMENT"))