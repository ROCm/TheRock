import os
import psycopg2
import argparse
# Load credentials from environment
host = 'therockcluster-1.cgwwfykpw9ix.us-east-2.redshift.amazonaws.com'
port = 5439
dbname = "workflow_database"
user = "awsuser"
password = "7mP+a[61{tIFDK!0"

parser = argparse.ArgumentParser(description="JSON input to parse")
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON input to populate the tables in DB",
    )

args = parser.parse_args()

print(args.input)

conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password,
    connect_timeout=60
)

cur = conn.cursor()

# Example insert - change table/values as needed
cur.execute("""
    INSERT INTO workflow_run_details (column1, column2) VALUES (%s, %s)
""", ("value1", "value2"))

conn.commit()
cur.close()
conn.close()