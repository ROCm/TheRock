import os
import psycopg2
import argparse
import json
# # Load credentials from environment
# host = 'therockcluster-1.cgwwfykpw9ix.us-east-2.redshift.amazonaws.com'
# port = 5439
# dbname = "workflow_database"
# user = "awsuser"
# password = "7mP+a[61{tIFDK!0"

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

# build_id = 306
# args.input = '{"total_count": 1, "jobs": [{"id": 44954385538, "run_id": 15935530212, "workflow_name": "Build Linux Packages", "head_branch": "users/arravikum/workflow_analysis", "run_url": "https://api.github.com/repos/ROCm/TheRock/actions/runs/15935530212", "run_attempt": 1, "node_id": "CR_kwDOLaI0488AAAAKd318gg", "head_sha": "a8978d7d0a8cd86e7c49a308cb92d119dac8f4e7", "url": "https://api.github.com/repos/ROCm/TheRock/actions/jobs/44954385539", "html_url": "https://github.com/ROCm/TheRock/actions/runs/15935530212/job/44954385539", "status": "in_progress", "conclusion": null, "created_at": "2025-06-27T20:39:11Z", "started_at": "2025-06-27T20:39:34Z", "completed_at": null, "name": "Build Linux Packages (xfail false)", "steps": [{"name": "Set up job", "status": "completed", "conclusion": "success", "number": 1, "started_at": "2025-06-27T20:39:35Z", "completed_at": "2025-06-27T20:39:36Z"}, {"name": "Initialize containers", "status": "completed", "conclusion": "success", "number": 2, "started_at": "2025-06-27T20:39:36Z", "completed_at": "2025-06-27T20:40:18Z"}, {"name": "Checking out repository", "status": "completed", "conclusion": "success", "number": 3, "started_at": "2025-06-27T20:40:18Z", "completed_at": "2025-06-27T20:40:19Z"}, {"name": "Configure AWS Credentials", "status": "completed", "conclusion": "success", "number": 4, "started_at": "2025-06-27T20:40:19Z", "completed_at": "2025-06-27T20:40:19Z"}, {"name": "Create Logs index Files", "status": "completed", "conclusion": "success", "number": 5, "started_at": "2025-06-27T20:40:19Z", "completed_at": "2025-06-27T20:40:19Z"}, {"name": "Determine job status for workflow run", "status": "in_progress", "conclusion": null, "number": 6, "started_at": "2025-06-27T20:40:19Z", "completed_at": null}, {"name": "Create job status JSON file for S3 upload", "status": "pending", "conclusion": null, "number": 7, "started_at": null, "completed_at": null}, {"name": "Install psycopg2", "status": "pending", "conclusion": null, "number": 8, "started_at": null, "completed_at": null}, {"name": "Populate tables in redshift", "status": "pending", "conclusion": null, "number": 9, "started_at": null, "completed_at": null}, {"name": "Upload Logs", "status": "pending", "conclusion": null, "number": 10, "started_at": null, "completed_at": null}, {"name": "Post Configure AWS Credentials", "status": "pending", "conclusion": null, "number": 18, "started_at": null, "completed_at": null}, {"name": "Post Checking out repository", "status": "pending", "conclusion": null, "number": 19, "started_at": null, "completed_at": null}, {"name": "Stop containers", "status": "pending", "conclusion": null, "number": 20, "started_at": null, "completed_at": null}], "check_run_url": "https://api.github.com/repos/ROCm/TheRock/check-runs/44954385538", "labels": ["azure-linux-scale-rocm"], "runner_id": 84105, "runner_name": "azure-linux-scale-rocm-76pqr-runner-x9dqp", "runner_group_id": 1, "runner_group_name": "default"}]}'
input_dict = json.loads(args.api_op)

print(args.input)

conn = psycopg2.connect(
    host=redshift_cluster_endpoint,
    port=redshift_port,
    dbname=dbname,
    user=redshift_username,
    password=redshift_password,
    connect_timeout=60
)

cur = conn.cursor()

# Example insert - change table/values as needed

cur.execute("Select * FROM workflow_run_details LIMIT 0")

colnames = [desc[0] for desc in cur.description]

cur.execute("Select * FROM step_status LIMIT 0")

colnames_steps = [desc[0] for desc in cur.description]


for i in range(0, len(input_dict['jobs'])):

    project = input_dict['jobs'][i]['run_url'].split("/")[5]

# workflow_run_details columns ['build_id', 'id', 'head_branch', 'workflow_name', 'project', 'started_at', 'run_url']

# step_status columns ['workflow_run_details_id', 'id', 'name', 'status', 'conclusion', 'started_at', 'completed_at']

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