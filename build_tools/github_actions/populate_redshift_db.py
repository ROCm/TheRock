"""
Populates the Redshift cluster with job status details.

This script is executed as part of the workflow after `fetch_job_status.py` completes.

Schema overview:
- Table: workflow_run_details
  Columns: ['run_id', 'id', 'head_branch', 'workflow_name', 'project', 'started_at', 'run_url']

- Table: step_status
  Columns: ['workflow_run_details_id', 'id', 'name', 'status', 'conclusion', 'started_at', 'completed_at']

"""

import logging
import os
import redshift_connector
import argparse
import json
import sys
import re

supported_workflows = ['Build Linux Packages']

logging.basicConfig(level=logging.INFO)


def populate_redshift_db(
    api_output,
    run_id,
    redshift_cluster_endpoint,
    dbname,
    redshift_username,
    redshift_password,
    redshift_port,
):
    logging.info(f"Github API output from Workflow {api_output}")

    input_dict = json.loads(api_output)

    logging.info("Starting Redshift metadata retrieval...")
    try:
        logging.info("Connecting to Redshift cluster...")
        with redshift_connector.connect(
            host=redshift_cluster_endpoint,
            port=redshift_port,
            database=dbname,
            user=redshift_username,
            password=redshift_password,
        ) as conn:
            with conn.cursor() as cursor:
                logging.info(
                    f"Successfully connected to Redshift"
                )

                try:
                    conn.autocommit = True

                    logging.info("Retrieving column metadata for 'workflow_run_details'...")
                    cursor.execute("SELECT * FROM workflow_run_details LIMIT 0")
                    colnames = [desc[0] for desc in cursor.description]
                    logging.info(
                        f"Retrieved {len(colnames)} columns from 'workflow_run_details': {colnames}"
                    )

                    logging.info("Retrieving column metadata for 'step_status'...")
                    cursor.execute("SELECT * FROM step_status LIMIT 0")
                    colnames_steps = [desc[0] for desc in cursor.description]
                    logging.info(
                        f"Retrieved {len(colnames_steps)} columns from 'step_status': {colnames_steps}"
                    )

                except Exception as e:
                    raise RuntimeError(f"Redshift metadata retreival failed: {e}")

                # Iterate over each job in the input dictionary
                for i in range(len(input_dict["jobs"])):
                    job = input_dict["jobs"][i]
                    """
                        Extract the project name from the GitHub Actions run URL.

                        The project name is located at the 6th segment (index 5) of the URL path.
                        Example:
                            For the URL "https://api.github.com/repos/ROCm/TheRock/actions/runs/16121346338",
                            the extracted project name will be "TheRock".
                    """
                    project = job["run_url"].split("/")[5]
                    """
                        Extract name of the platforms job is run on name from the GitHub API output.

                        The platform name is located inside the parentheses of the value in input_dict["jobs"][i]["name"]
                        Example:
                            For the job - input_dict['jobs'][6]['name'] output will be as below,
                            'Linux (linux-mi300-1gpu-ossci-rocm, gfx94X-dcgpu, gfx942) / Build / Build Linux Packages (xfail false)''
                            the extracted project name will be 'gfx94X-dcgpu, gfx942'.
                    """
                    platform_str = input_dict['jobs'][i]['name']
                    if 'gfx' in platform_str:
                        # Extract first (...) group to filter out GPUs 
                        match = re.search(r"\(([^)]*)\)", platform_str)
                        inside = match.group(1) if match else ""
                        # Split by comma and filter for entries starting with 'gfx'
                        gpu_list = [item.strip() for item in inside.split(",") if item.strip().startswith("gfx")]
                        platform = ", ".join(gpu_list)
                    else:
                        platform = ""
                    match_job = re.search(r"[^/]+$", platform_str)
                    workflow_id = job["id"]
                    head_branch = job["head_branch"]
                    workflow_name = job["workflow_name"]
                    workflow_job_name = match_job.group(0).lstrip()
                    workflow_started_at = job["started_at"]
                    run_url = job["run_url"]
                    if platform != "":
                        logging.info(
                            f"Inserting workflow run details into 'workflow_run_details' table: "
                            f"run_id={run_id}, id={workflow_id}, workflow_job_name={workflow_job_name}, head_branch='{head_branch}', "
                            f"workflow_name='{workflow_name}', platform='{platform}', project='{project}', "
                            f"started_at='{workflow_started_at}', run_url='{run_url}'"
                        )

                        # Insert workflow run details into the database

                        cursor.execute(
                            """
                                INSERT INTO workflow_run_details
                                    ("run_id", "id", "head_branch", "workflow_name", "workflow_job_name", "platform", "project", "started_at", "run_url")
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                run_id,
                                workflow_id,
                                head_branch,
                                workflow_name,
                                workflow_job_name,
                                platform,
                                project,
                                workflow_started_at,
                                run_url,
                            ),
                        )

                        # Iterate over each step in the current job
                        for j in range(len(job["steps"])):
                            step = job["steps"][j]

                            steps_id = job["id"]
                            steps_name = step["name"]
                            status = step["status"]
                            conclusion = step["conclusion"]
                            step_started_at = step["started_at"]
                            step_completed_at = step["completed_at"]

                            logging.info(
                                f"Inserting step status into 'step_status' table: "
                                f"workflow_run_details_id={steps_id}, id={j + 1}, name='{steps_name}', "
                                f"status='{status}', conclusion='{conclusion}', "
                                f"started_at='{step_started_at}', completed_at='{step_completed_at}"
                            )

                            # Insert step status details into the database
                            # j + 1 is used to populate an ID for each step in the each workflow run
                            cursor.execute(
                                """
                                    INSERT INTO step_status
                                        ("workflow_run_details_id", "id", "name", "status", "conclusion", "started_at", "completed_at")
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    steps_id,
                                    j + 1,
                                    steps_name,
                                    status,
                                    conclusion,
                                    step_started_at,
                                    step_completed_at,
                                ),
                            )

    except Exception as e:
        raise RuntimeError(f"Redshift connection failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Populate DB in redshift cluster")
    parser.add_argument(
        "--api-output",
        type=str,
        help="GitHub API output to populate the tables in DB",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="run_id to populate the tables in DB",
    )
    parser.add_argument(
        "--redshift-cluster-endpoint",
        type=str,
        help="redshift cluster endpoint to access cluster",
    )
    parser.add_argument(
        "--dbname",
        type=str,
        help="database name to populate the tables in DB",
    )
    parser.add_argument(
        "--redshift-username",
        type=str,
        help="username to access redshift cluster",
    )
    parser.add_argument(
        "--redshift-password",
        type=str,
        help="password to access redshift cluster",
    )
    parser.add_argument(
        "--redshift-port",
        type=int,
        default=5439,
        help="port to access redshift cluster",
    )
    args = parser.parse_args()

    populate_redshift_db(
        args.api_output,
        args.run_id,
        args.redshift_cluster_endpoint,
        args.dbname,
        args.redshift_username,
        args.redshift_password,
        args.redshift_port,
    )


if __name__ == "__main__":
    main()
