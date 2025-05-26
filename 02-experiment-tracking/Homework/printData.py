import sqlite3
import pandas as pd

# Connect to the MLflow database
conn = sqlite3.connect("mlflow.db")

# Query data (example: list all experiments)
query = "SELECT * FROM latest_metrics"
df = pd.read_sql_query(query, conn)

# Display the data in tabular format
print(df.to_string(index=False))

# Close the connection
conn.close()
