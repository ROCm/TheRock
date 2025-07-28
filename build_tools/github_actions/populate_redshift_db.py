"""
Populates the Redshift cluster with job status details.

This script is executed as part of the workflow after `fetch_job_status.py` completes.

Schema overview:
- Table: workflow_run_details
  Columns: ['run_id', 'id', 'head_branch', 'workflow_name', 'workflow_job_name','platform' 'project', 'started_at', 'run_url']

- Table: step_status
  Columns: ['workflow_job_id', 'id', 'name', 'status', 'conclusion', 'started_at', 'completed_at']

"""

import logging
import os
import redshift_connector
import argparse
import json
import sys
import re

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
    # Validate presence of required Redshift connection parameters and collect any missing ones
    missing = []
    if not redshift_cluster_endpoint:
        missing.append("--redshift-cluster-endpoint")
    if not dbname:
        missing.append("--dbname")
    if not redshift_username:
        missing.append("--redshift-username")
    if not redshift_password:
        missing.append("--redshift-password")
    if not redshift_port:
        missing.append("--redshift-port")

    # Fail gracefully if required parameters are missing
    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}")
        sys.exit(1)

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
                logging.info(f"Successfully connected to Redshift")
                """
                Retrieve column metadata from Redshift tables:

                - Enable autocommit mode.
                - Use SELECT ... LIMIT 0 to fetch only schema without data.
                - Log column names for 'workflow_run_details' and 'step_status'.
                - Raise RuntimeError if retrieval fails.
                """
                try:
                    conn.autocommit = True

                    logging.info(
                        "Retrieving column metadata for 'workflow_run_details'..."
                    )
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
                    Extract the GPU architecture identifiers from the GitHub Actions job name.

                    The platform string often includes multiple comma-separated values within parentheses,
                    where GPU identifiers typically start with 'gfx'. We extract those values into a list.

                    Example:
                    Given input_dict['jobs'][6]['name'] as:
                    'Linux (linux-mi325-1gpu-ossci-rocm, gfx94X-dcgpu, gfx942) / Build / Build Linux Packages (xfail false)'

                    This will extract:
                    ['gfx94X-dcgpu', 'gfx942']
                    """
                    platform_str = input_dict["jobs"][i]["name"]
                    if "gfx" in platform_str:
                        # Extract first group to filter out GPUs from platform_str Linux (linux-mi325-1gpu-ossci-rocm, gfx94X-dcgpu, gfx942) / Build / Build Linux Packages (xfail false) into list [gfx94X-dcgpu, gfx942]
                        match = re.search(r"\(([^)]*)\)", platform_str)
                        inside = match.group(1) if match else ""
                        # Split by comma and filter for entries starting with 'gfx'
                        gpu_list = [
                            item.strip()
                            for item in inside.split(",")
                            if item.strip().startswith("gfx")
                        ]

                        platform = ", ".join(gpu_list)
                    else:
                        platform = ""
                    match_job = re.search(r"[^/]+$", platform_str)
                    job_id = int(job["id"])
                    head_branch = job["head_branch"]
                    workflow_name = job["workflow_name"]
                    workflow_job_name = match_job.group(0).lstrip()
                    workflow_started_at = job["started_at"]
                    run_url = job["run_url"]

                    logging.info(
                        f"Inserting workflow run details into 'workflow_run_details' table: "
                        f"run_id={run_id}, id={job_id}, workflow_job_name={workflow_job_name}, head_branch='{head_branch}', "
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
                            job_id,
                            head_branch,
                            workflow_name,
                            workflow_job_name,
                            platform,
                            project,
                            workflow_started_at,
                            run_url,
                        ),
                    )
                    logging.info(
                        f"Inserting 'step_status' table: "
                        f"workflow_job_id={job['id']}"
                    )
                    # Iterate over each step in the current job
                    for j in range(len(job["steps"])):
                        step = job["steps"][j]
                        job_id = job["id"]
                        steps_name = step["name"]
                        status = step["status"]
                        conclusion = step["conclusion"]
                        step_started_at = step["started_at"]
                        step_completed_at = step["completed_at"]

                        logging.info(
                            f"Inserting workflow_job_id={job_id}, name='{steps_name}', "
                            f"status='{status}', conclusion='{conclusion}' "
                        )

                        # Insert step status details into the database
                        # j + 1 is used to populate an ID for each step in the each workflow run
                        cursor.execute(
                            """
                                INSERT INTO step_status
                                    ("workflow_job_id", "id", "name", "status", "conclusion", "started_at", "completed_at")
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                job_id,
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
        required=True,
        help="GitHub API output to populate the tables in DB",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="run_id to populate the tables in DB",
    )
    parser.add_argument(
        "--redshift-cluster-endpoint",
        type=str,
        required=True,
        help="redshift cluster endpoint to access cluster",
    )
    parser.add_argument(
        "--dbname",
        type=str,
        required=True,
        help="database name to populate the tables in DB",
    )
    parser.add_argument(
        "--redshift-username",
        type=str,
        required=True,
        help="username to access redshift cluster",
    )
    parser.add_argument(
        "--redshift-password",
        type=str,
        required=True,
        help="password to access redshift cluster",
    )
    parser.add_argument(
        "--redshift-port",
        type=int,
        required=True,
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
