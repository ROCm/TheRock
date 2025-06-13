#!/usr/bin/env python
# Populates Redshift cluster DB for job status details.
# This script is incorporated as part of the workflow
# after fetch_job_status.py is run
# Tables and column details below
# workflow_run_details columns ['build_id', 'id', 'head_branch', 'workflow_name', 'project', 'started_at', 'run_url']
# step_status columns ['workflow_run_details_id', 'id', 'name', 'status', 'conclusion', 'started_at', 'completed_at']

import os
import psycopg2
import argparse
import json

parser = argparse.ArgumentParser(description="Populate DB in redshift cluster")
parser.add_argument(
        "--api_op",
        type=str,
        help="github API output to populate the tables in DB",
    )
parser.add_argument(
        "--build_id",
        type=int,
        help="github action build_id to populate the tables in DB",
    )
parser.add_argument(
        "--redshift_cluster_endpoint",
        type=str,
        help="github action redshift cluster endpoint to access cluster",
    )
parser.add_argument(
        "--dbname",
        type=str,
        help="github action database name to populate the tables in DB",
    )
parser.add_argument(
        "--redshift_username",
        type=str,
        help="github action awsuser name name to access redshift cluster",
    )

parser.add_argument(
        "--redshift_password",
        type=str,
        help="github action awsuser password to access redshift cluster",
    )
parser.add_argument(
        "--redshift_port",
        type=int, default=5439,
        help="port to access redshift cluster",
        )

args = parser.parse_args()

build_id = args.build_id

input_dict = json.loads(args.api_op)

print(args.api_op)

conn = psycopg2.connect(
    host=args.redshift_cluster_endpoint,
    port=args.redshift_port,
    dbname=args.dbname,
    user=args.redshift_username,
    password=args.redshift_password,
    connect_timeout=60
)

cur = conn.cursor()

cur.execute("Select * FROM workflow_run_details LIMIT 0")

colnames = [desc[0] for desc in cur.description]

cur.execute("Select * FROM step_status LIMIT 0")

colnames_steps = [desc[0] for desc in cur.description]


for i in range(0, len(input_dict['jobs'])):

    project = input_dict['jobs'][i]['run_url'].split("/")[5]
    cur.execute("""
        INSERT INTO workflow_run_details ("build_id", "id", "head_branch", "workflow_name", "project", "started_at", "run_url") VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (build_id, input_dict['jobs'][i]['id'], input_dict['jobs'][i]['head_branch'], input_dict['jobs'][i]['workflow_name'], project, input_dict['jobs'][i]['started_at'], input_dict['jobs'][i]['run_url'] ))

    for j in range(0, len(input_dict['jobs'][i]['steps'])):
        cur.execute("""
        INSERT INTO step_status ("workflow_run_details_id", "id", "name", "status", "conclusion", "started_at", "completed_at") VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (input_dict['jobs'][i]['id'], int(j)+1, input_dict['jobs'][i]['steps'][j]['name'], input_dict['jobs'][i]['steps'][j]['status'], input_dict['jobs'][i]['steps'][j]['conclusion'], input_dict['jobs'][i]['steps'][j]['started_at'], input_dict['jobs'][i]['steps'][j]['completed_at'] ))



conn.commit()
cur.close()
conn.close()