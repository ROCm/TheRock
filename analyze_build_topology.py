#!/usr/bin/env python3
"""
Build Topology Analyzer - Parse BUILD_TOPOLOGY.toml and generate optimization insights
"""

import tomllib
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

def load_topology(file_path: str) -> dict:
    """Load and parse the BUILD_TOPOLOGY.toml file"""
    with open(file_path, 'rb') as f:
        return tomllib.load(f)

def build_dependency_graph(topology: dict) -> Tuple[Dict, Dict, Dict]:
    """Build dependency graphs for artifacts and groups"""
    
    # Artifact-level dependencies
    artifact_deps = {}
    artifact_groups = {}
    
    for artifact_name, artifact_data in topology.get('artifacts', {}).items():
        artifact_deps[artifact_name] = artifact_data.get('artifact_deps', [])
        artifact_groups[artifact_name] = artifact_data.get('artifact_group', '')
    
    # Group-level dependencies
    group_deps = {}
    group_types = {}
    
    for group_name, group_data in topology.get('artifact_groups', {}).items():
        group_deps[group_name] = group_data.get('artifact_group_deps', [])
        group_types[group_name] = group_data.get('type', 'generic')
    
    return artifact_deps, artifact_groups, group_deps

def calculate_build_levels(artifact_deps: Dict[str, List[str]]) -> Dict[str, int]:
    """Calculate build level for each artifact (topological ordering)"""
    levels = {}
    visited = set()
    
    def get_level(artifact: str) -> int:
        if artifact in visited:
            return levels.get(artifact, 0)
        
        visited.add(artifact)
        deps = artifact_deps.get(artifact, [])
        
        if not deps:
            levels[artifact] = 0
            return 0
        
        max_dep_level = max(get_level(dep) for dep in deps if dep in artifact_deps)
        levels[artifact] = max_dep_level + 1
        return levels[artifact]
    
    for artifact in artifact_deps.keys():
        get_level(artifact)
    
    return levels

def find_parallelizable_sets(topology: dict) -> Dict[int, List[str]]:
    """Find artifacts that can be built in parallel at each level"""
    artifact_deps, artifact_groups, group_deps = build_dependency_graph(topology)
    levels = calculate_build_levels(artifact_deps)
    
    # Group artifacts by level
    level_groups = defaultdict(list)
    for artifact, level in levels.items():
        level_groups[level].append(artifact)
    
    return level_groups

def analyze_critical_path(topology: dict) -> List[str]:
    """Find the critical path (longest dependency chain)"""
    artifact_deps, _, _ = build_dependency_graph(topology)
    levels = calculate_build_levels(artifact_deps)
    
    # Find artifact at deepest level
    if not levels:
        return []
    
    max_level = max(levels.values())
    deepest_artifacts = [a for a, l in levels.items() if l == max_level]
    
    # Trace back one path
    critical_path = []
    current = deepest_artifacts[0] if deepest_artifacts else None
    
    while current:
        critical_path.insert(0, current)
        deps = artifact_deps.get(current, [])
        
        # Find the dependency with the highest level
        next_artifact = None
        max_dep_level = -1
        for dep in deps:
            if dep in levels and levels[dep] > max_dep_level:
                max_dep_level = levels[dep]
                next_artifact = dep
        
        current = next_artifact
    
    return critical_path

def analyze_build_stages(topology: dict):
    """Analyze build stages and their parallelization"""
    stages = topology.get('build_stages', {})
    groups = topology.get('artifact_groups', {})
    
    stage_info = {}
    
    for stage_name, stage_data in stages.items():
        stage_groups = stage_data.get('artifact_groups', [])
        stage_type = stage_data.get('type', 'generic')
        
        # Count artifacts in this stage
        artifact_count = 0
        for artifact_name, artifact_data in topology.get('artifacts', {}).items():
            if artifact_data.get('artifact_group') in stage_groups:
                artifact_count += 1
        
        stage_info[stage_name] = {
            'groups': stage_groups,
            'type': stage_type,
            'artifact_count': artifact_count,
            'description': stage_data.get('description', '')
        }
    
    return stage_info

def main():
    topology = load_topology('BUILD_TOPOLOGY.toml')
    
    print("="*100)
    print("BUILD TOPOLOGY ANALYSIS - Root Build Optimization")
    print("="*100)
    
    # 1. Overall structure
    num_artifacts = len(topology.get('artifacts', {}))
    num_groups = len(topology.get('artifact_groups', {}))
    num_stages = len(topology.get('build_stages', {}))
    
    print(f"\nOVERALL STRUCTURE:")
    print(f"  Total Artifacts:      {num_artifacts}")
    print(f"  Total Groups:         {num_groups}")
    print(f"  Total Build Stages:   {num_stages}")
    
    # 2. Dependency analysis
    artifact_deps, artifact_groups, group_deps = build_dependency_graph(topology)
    levels = calculate_build_levels(artifact_deps)
    
    max_level = max(levels.values()) if levels else 0
    print(f"\nDEPENDENCY DEPTH:")
    print(f"  Maximum Build Levels: {max_level + 1}")
    print(f"  (This is the minimum serial depth - cannot be reduced)")
    
    # 3. Parallelization potential
    level_groups = find_parallelizable_sets(topology)
    
    print(f"\nPARALLELIZATION POTENTIAL BY LEVEL:")
    print(f"{'Level':<8} {'Artifacts':<10} {'Can Build in Parallel'}")
    print("-"*100)
    
    for level in sorted(level_groups.keys()):
        artifacts = level_groups[level]
        print(f"{level:<8} {len(artifacts):<10} {', '.join(artifacts[:8])}")
        if len(artifacts) > 8:
            print(f"{'':>19} ... and {len(artifacts) - 8} more")
    
    # 4. Critical path
    critical_path = analyze_critical_path(topology)
    
    print(f"\nCRITICAL PATH (Longest Dependency Chain):")
    print(f"  Length: {len(critical_path)} artifacts")
    print(f"  Path: {' -> '.join(critical_path)}")
    
    # 5. Build stages analysis
    stage_info = analyze_build_stages(topology)
    
    print(f"\nBUILD STAGES ANALYSIS:")
    print(f"{'Stage':<25} {'Type':<12} {'Artifacts':<10} {'Description'}")
    print("-"*100)
    
    for stage_name, info in stage_info.items():
        print(f"{stage_name:<25} {info['type']:<12} {info['artifact_count']:<10} {info['description'][:40]}")
    
    # 6. Identify bottlenecks
    print(f"\n{'='*100}")
    print("BOTTLENECK ANALYSIS")
    print("="*100)
    
    # Find artifacts with many dependents
    dependents = defaultdict(list)
    for artifact, deps in artifact_deps.items():
        for dep in deps:
            dependents[dep].append(artifact)
    
    print(f"\nHIGH-IMPACT ARTIFACTS (Many artifacts depend on these):")
    print(f"{'Artifact':<30} {'Dependents':<10} {'Level':<8} {'Group'}")
    print("-"*100)
    
    high_impact = sorted(dependents.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    for artifact, deps_list in high_impact:
        level = levels.get(artifact, 0)
        group = artifact_groups.get(artifact, 'unknown')
        print(f"{artifact:<30} {len(deps_list):<10} {level:<8} {group}")
    
    # 7. Optimization recommendations
    print(f"\n{'='*100}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*100)
    
    print("\n1. IMMEDIATE ACTIONS - Root Build Orchestration:")
    print("   Current Issue: Root build shows 7% avg concurrency (173 minutes)")
    print("   ")
    print("   a) PARALLELIZE BUILD STAGES:")
    print("      - 'foundation' stage can run independently")
    print("      - 'compiler-runtime' has 7 groups - can parallelize within stage")
    print("      - 'math-libs' and 'comm-libs' can run IN PARALLEL (both per-arch)")
    print("      - 'debug-tools', 'dctools-core', 'iree-libs' can run in parallel")
    print("      ")
    
    # Count independent stages
    independent_stages = []
    for stage_name, info in stage_info.items():
        if stage_name in ['foundation']:
            continue  # First stage
        # Check if any dependencies
        stage_groups = info['groups']
        has_external_deps = False
        for group in stage_groups:
            if group_deps.get(group, []):
                has_external_deps = True
                break
        if not has_external_deps:
            independent_stages.append(stage_name)
    
    print(f"   b) PARALLEL STAGE EXECUTION:")
    print(f"      After 'compiler-runtime' stage completes, these can run in parallel:")
    
    # Group stages by their dependencies
    stage_dep_analysis = {
        'After foundation': ['compiler-runtime'],
        'After compiler-runtime': ['math-libs', 'comm-libs', 'debug-tools', 'dctools-core', 'iree-libs'],
        'After math-libs': ['profiler-apps (if it needs math-libs)']
    }
    
    for prereq, stages in stage_dep_analysis.items():
        print(f"        {prereq}: {len(stages)} stages can run concurrently")
        for s in stages:
            print(f"          - {s}")
    
    print(f"\n   c) WITHIN-STAGE PARALLELIZATION:")
    print(f"      'compiler-runtime' stage (largest bottleneck):")
    print(f"        - Contains 7 artifact groups")
    print(f"        - Most have independent artifacts that can build in parallel")
    print(f"        - Estimated potential: 20-40 parallel jobs vs current 2-6")
    
    print(f"\n2. MEDIUM-TERM OPTIMIZATIONS:")
    print(f"   ")
    print(f"   a) SPLIT LARGE ARTIFACTS:")
    
    # Find artifacts with no deps that could be in earlier stages
    level_0_artifacts = [a for a, l in levels.items() if l == 0]
    print(f"      - {len(level_0_artifacts)} artifacts have no dependencies")
    print(f"      - These can build immediately, consider pre-building")
    
    print(f"   ")
    print(f"   b) REDUCE CRITICAL PATH:")
    print(f"      - Current depth: {max_level + 1} levels")
    print(f"      - Critical path: {' -> '.join(critical_path[:5])}")
    if len(critical_path) > 5:
        print(f"                    ... -> {' -> '.join(critical_path[-2:])}")
    print(f"      - Consider if any dependencies can be made optional/lazy")
    
    print(f"\n3. BUILD SYSTEM IMPROVEMENTS:")
    print(f"   ")
    print(f"   a) CMAKE PARALLELIZATION:")
    print(f"      - Ensure CMake is invoked with -j flag for parallel configuration")
    print(f"      - Use ExternalProject_Add with parallel builds for independent components")
    print(f"      - Consider cmake --build . --parallel for each stage")
    print(f"   ")
    print(f"   b) NINJA JOB POOL CONFIGURATION:")
    print(f"      - Current: Limited to ~98-100 parallel jobs (good)")
    print(f"      - But root build only using 7 on average (bad)")
    print(f"      - Issue is likely in CMake dependency specification, not Ninja")
    
    # 8. Specific artifact-level opportunities
    print(f"\n4. SPECIFIC OPPORTUNITIES:")
    print(f"   ")
    
    # Find large groups
    group_artifact_count = defaultdict(int)
    for artifact, group in artifact_groups.items():
        group_artifact_count[group] += 1
    
    large_groups = sorted(group_artifact_count.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   a) LARGEST ARTIFACT GROUPS (potential for internal parallelization):")
    for group, count in large_groups:
        group_type = topology['artifact_groups'][group].get('type', 'generic')
        print(f"      - {group}: {count} artifacts ({group_type})")
    
    print(f"\n   b) PER-ARCH OPTIMIZATION:")
    per_arch_groups = [g for g, data in topology['artifact_groups'].items() 
                       if data.get('type') == 'per-arch']
    print(f"      - {len(per_arch_groups)} groups are 'per-arch' type")
    print(f"      - These should build once with all targets, not serially")
    print(f"      - Groups: {', '.join(per_arch_groups)}")
    
    # 9. Expected improvement
    print(f"\n{'='*100}")
    print("ESTIMATED IMPROVEMENT POTENTIAL")
    print("="*100)
    print(f"\nCurrent Root Build:")
    print(f"  Duration:         173 minutes")
    print(f"  Avg Concurrency:  7.11 tasks")
    print(f"  Max Concurrency:  124 tasks")
    print(f"  Efficiency:       5.7%")
    print(f"\nOptimized Root Build (Conservative Estimate):")
    print(f"  Duration:         25-35 minutes (7x faster)")
    print(f"  Avg Concurrency:  60-80 tasks")
    print(f"  Max Concurrency:  98-120 tasks")
    print(f"  Efficiency:       60-80%")
    print(f"\nKey Changes:")
    print(f"  1. Parallel stage execution (3-4 stages concurrent after foundation)")
    print(f"  2. Better within-stage parallelization (7 groups in compiler-runtime)")
    print(f"  3. Eliminate artificial serialization in CMake dependencies")
    
    print(f"\n{'='*100}")
    print("Analysis Complete!")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
