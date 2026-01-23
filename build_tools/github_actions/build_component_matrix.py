#!/usr/bin/env python3
"""
This script builds a flattened component × shard matrix for parallel test execution.

Takes the output from fetch_test_configurations.py (component list with shard info)
and flattens it into individual matrix items (one per component-shard combination).

Required environment variables:
  - COMPONENTS_JSON: JSON string containing the components array from fetch_test_configurations.py

Outputs to GITHUB_OUTPUT:
  - component_matrix: Flattened JSON array of component-shard combinations
"""

import json
import os
import sys
from github_actions_utils import gha_set_output


def build_matrix(components_json: str) -> list[dict]:
    """
    Build a flat matrix with component + shard combinations.
    
    Args:
        components_json: JSON string of components from fetch_test_configurations.py
        
    Returns:
        List of matrix items, each containing component info and specific shard
    """
    components = json.loads(components_json)
    
    matrix_items = []
    for component in components:
        total_shards = component.get('total_shards', 1)
        shard_arr = component.get('shard_arr', [1])
        job_name = component['job_name']
        
        for shard in shard_arr:
            # Create display name for this specific test
            if total_shards > 1:
                display_name = f"{job_name} (shard {shard}/{total_shards})"
            else:
                display_name = job_name
            
            matrix_items.append({
                'component': component,
                'shard': shard,
                'display_name': display_name
            })
    
    return matrix_items


def run():
    components_json = os.environ.get('COMPONENTS_JSON', '[]')
    
    if not components_json or components_json == '[]':
        print("No components to process, outputting empty matrix")
        gha_set_output({'component_matrix': '[]'})
        return
    
    matrix_items = build_matrix(components_json)
    matrix_json = json.dumps(matrix_items)
    
    print(f"Generated matrix with {len(matrix_items)} items:")
    for item in matrix_items:
        print(f"  - {item['display_name']}")
    
    gha_set_output({'component_matrix': matrix_json})


if __name__ == "__main__":
    run()
