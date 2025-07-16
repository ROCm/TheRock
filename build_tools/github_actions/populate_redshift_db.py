"""
Populates the Redshift cluster with job status details.

This script is executed as part of the workflow after `fetch_job_status.py` completes.

Schema overview:
- Table: workflow_run_details  
  Columns: ['build_id', 'id', 'head_branch', 'workflow_name', 'project', 'started_at', 'run_url']

- Table: step_status  
  Columns: ['workflow_run_details_id', 'id', 'name', 'status', 'conclusion', 'started_at', 'completed_at']
"""

import logging
import os
import redshift_connector
import argparse
import json
import sys


logging.basicConfig(level=logging.INFO)
        
def populate_redshift_db(log, api_output_dict, build_id, redshift_cluster_endpoint, dbname, redshift_username, redshift_password, redshift_port):

        log.info(f"Github API output from Workflow (truncated): {str(api_output_dict)[:500]}...")

        input_dict = api_output_dict

        log.info("Starting Redshift metadata retrieval...")

        try:
            log.info("Connecting to Redshift cluster...")
            conn = redshift_connector.connect( 
                host=redshift_cluster_endpoint,
                port=int(redshift_port),
                database=dbname,
                user=redshift_username,
                password=redshift_password
            )
            log.info(f"Successfully connected to Redshift at {redshift_cluster_endpoint}:{redshift_port}")
        except Exception as e:
            log.error(f"Failed to connect to Redshift: {e}")
            sys.exit(1)

        try:
            cur = conn.cursor()
            conn.autocommit = True
            
            log.info("Retrieving column metadata for 'workflow_run_details'...")
            cur.execute("SELECT * FROM workflow_run_details LIMIT 0")
            colnames = [desc[0] for desc in cur.description]
            log.info(f"Retrieved {len(colnames)} columns from 'workflow_run_details': {colnames}")

            log.info("Retrieving column metadata for 'step_status'...")
            cur.execute("SELECT * FROM step_status LIMIT 0")
            colnames_steps = [desc[0] for desc in cur.description]
            log.info(f"Retrieved {len(colnames_steps)} columns from 'step_status': {colnames_steps}")

        except Exception as e:
            log.error(f"Error during metadata retrieval: {e}")
            conn.close()
            sys.exit(1)


        # Iterate over each job in the input dictionary
        for i in range(len(input_dict['jobs'])):
            job = input_dict['jobs'][i]
            project = job['run_url'].split("/")[5]

            workflow_id = job['id']
            head_branch = job['head_branch']
            workflow_name = job['workflow_name']
            workflow_started_at = job['started_at']
            run_url = job['run_url']

            logging.info(
                f"Inserting workflow run details into 'workflow_run_details' table: "
                f"build_id={build_id}, id={workflow_id}, head_branch='{head_branch}', "
                f"workflow_name='{workflow_name}', project='{project}', "
                f"started_at='{workflow_started_at}', run_url='{run_url}'"
            )

            cur.execute("""
                INSERT INTO workflow_run_details 
                    ("build_id", "id", "head_branch", "workflow_name", "project", "started_at", "run_url") 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (build_id, workflow_id, head_branch, workflow_name, project, workflow_started_at, run_url))

            for j in range(len(job['steps'])):
                step = job['steps'][j]

                steps_id = job['id']
                steps_name = step['name']
                status = step['status']
                conclusion = step['conclusion']
                step_started_at = step['started_at']
                step_completed_at = step['completed_at']

                logging.info(
                    f"Inserting step status into 'step_status' table: "
                    f"workflow_run_details_id={steps_id}, id={j + 1}, name='{steps_name}', "
                    f"status='{status}', conclusion='{conclusion}', "
                    f"started_at='{step_started_at}', completed_at='{step_completed_at}'"
                )

                cur.execute("""
                    INSERT INTO step_status 
                        ("workflow_run_details_id", "id", "name", "status", "conclusion", "started_at", "completed_at") 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (steps_id, j + 1, steps_name, status, conclusion, step_started_at, step_completed_at))
 
        if cur:
            cur.close()
        if conn:
            conn.close()


def main():
        parser = argparse.ArgumentParser(description="Populate DB in redshift cluster")
        parser.add_argument(
                "--api-output-file",
                type=str,
                required=True,
                help="Path to file containing GitHub API output to populate the tables in DB",
            )
        parser.add_argument("--build-id", type=int, required=True)
        parser.add_argument("--redshift-cluster-endpoint", type=str, required=True)
        parser.add_argument("--dbname", type=str, required=True)
        parser.add_argument("--redshift-username", type=str, required=True)
        parser.add_argument("--redshift-password", type=str, required=True)
        parser.add_argument("--redshift-port", type=int, default=5439)

        args = parser.parse_args()

        try:
            with open(args.api_output_file, 'r') as f:
                api_output_dict = json.load(f)
        except Exception as e:
            logging.error(f"Failed to read API output file: {e}")
            sys.exit(1)

        populate_redshift_db(
            logging,
            api_output_dict,
            args.build_id,
            args.redshift_cluster_endpoint,
            args.dbname,
            args.redshift_username,
            args.redshift_password,
            args.redshift_port
        )


if __name__ == "__main__":
    main()
