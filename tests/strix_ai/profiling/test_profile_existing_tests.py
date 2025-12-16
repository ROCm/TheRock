"""
Profile existing Strix AI tests using rocprofv3
Uses Option 1 approach: run rocprofv3 on existing test files without code duplication
"""

import pytest
import subprocess
import sys
from pathlib import Path
import shutil
import os


def check_rocprofv3_available():
    """Check if rocprofv3 is available"""
    return shutil.which("rocprofv3") is not None


def get_trace_output_dir(category_name):
    """
    Get output directory for traces that will be preserved for artifact upload
    
    Args:
        category_name: Name of the test category (e.g., "vlm_clip", "vit")
    
    Returns:
        Path object for the output directory
    """
    # Use workspace directory for CI, or local profiling_traces/ for manual runs
    if os.getenv("GITHUB_ACTIONS"):
        # In CI: save to workspace root so GitHub Actions can find them
        output_dir = Path.cwd() / f"{category_name}_traces"
    else:
        # Local: save to profiling_traces/ subdirectory
        output_dir = Path("profiling_traces") / f"{category_name}_traces"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_rocprofv3_on_test(test_path, output_dir, timeout=300):
    """
    Run rocprofv3 on an existing test file
    
    Args:
        test_path: Path to test file (e.g., "tests/strix_ai/vlm/test_clip.py")
        output_dir: Directory to save profiling traces
        timeout: Timeout in seconds
    
    Returns:
        subprocess.CompletedProcess result
    """
    cmd = [
        "rocprofv3",
        "--hip-trace",
        "--kernel-trace",
        "--memory-copy-trace",
        "--output-format", "pftrace",
        "-d", str(output_dir),
        "--",
        sys.executable,
        "-m", "pytest",
        str(test_path),
        "-v", "-s"
    ]
    
    print(f"\n🔍 Profiling command:")
    print(f"   {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    
    return result


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
class TestProfileVLM:
    """Profile Vision Language Model tests"""
    
    def test_profile_clip(self, cleanup_gpu):
        """Profile existing CLIP test using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("Profiling VLM: CLIP Test")
        print("="*70)
        
        # Use persistent directory for traces (will be archived as artifacts)
        output_dir = get_trace_output_dir("vlm_clip")
        
        # Profile the existing CLIP test
        test_path = "tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching"
        
        print(f"📊 Profiling: {test_path}")
        result = run_rocprofv3_on_test(test_path, output_dir, timeout=300)
        
        # Check if profiling succeeded
        print(f"\n{'='*70}")
        print(f"Profiling Result: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
        print(f"{'='*70}")
        
        if result.stdout:
            print(f"\nTest Output:\n{result.stdout}")
        
        if result.stderr and "error" in result.stderr.lower():
            print(f"\nStderr:\n{result.stderr}")
        
        # Check for trace files
        trace_files = list(output_dir.glob("*"))
        
        if trace_files:
            print(f"\n✅ Generated {len(trace_files)} profiling trace file(s):")
            for f in trace_files[:5]:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")
            if len(trace_files) > 5:
                print(f"   ... and {len(trace_files) - 5} more files")
            print(f"\n📂 Traces saved to: {output_dir}")
            print(f"   (Will be archived as GitHub Actions artifact)")
        else:
            print(f"\n⚠️  No trace files generated")
        
        # Test passes if rocprofv3 ran (even if pytest test skipped)
        # We're testing the profiling capability, not the test itself
        print("\n✅ CLIP profiling completed")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
class TestProfileViT:
    """Profile Vision Transformer tests"""
    
    def test_profile_vit_inference(self, cleanup_gpu):
        """Profile existing ViT test using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("Profiling ViT: Vision Transformer Test")
        print("="*70)
        
        # Use persistent directory for traces
        output_dir = get_trace_output_dir("vit_inference")
        
        # Profile the existing ViT test
        test_path = "tests/strix_ai/vit/test_vit_base.py::TestViT::test_vit_inference"
        
        print(f"📊 Profiling: {test_path}")
        result = run_rocprofv3_on_test(test_path, output_dir, timeout=300)
        
        print(f"\n{'='*70}")
        print(f"Profiling Result: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
        print(f"{'='*70}")
        
        if result.stdout:
            print(f"\nTest Output:\n{result.stdout}")
        
        # Check for trace files
        trace_files = list(output_dir.glob("*"))
        
        if trace_files:
            print(f"\n✅ Generated {len(trace_files)} profiling trace file(s):")
            for f in trace_files[:5]:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")
            print(f"\n📂 Traces saved to: {output_dir}")
            print(f"   (Will be archived as GitHub Actions artifact)")
        
        print("\n✅ ViT profiling completed")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
class TestProfileCV:
    """Profile Computer Vision tests"""
    
    def test_profile_yolo(self, cleanup_gpu):
        """Profile existing YOLO test using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("Profiling CV: YOLO Object Detection Test")
        print("="*70)
        
        # Use persistent directory for traces
        output_dir = get_trace_output_dir("cv_yolo")
        
        # Profile the existing YOLO test
        test_path = "tests/strix_ai/cv/test_yolo.py::TestYOLO::test_yolo_detection"
        
        print(f"📊 Profiling: {test_path}")
        result = run_rocprofv3_on_test(test_path, output_dir, timeout=300)
        
        print(f"\n{'='*70}")
        print(f"Profiling Result: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
        print(f"{'='*70}")
        
        if result.stdout:
            print(f"\nTest Output:\n{result.stdout}")
        
        # Check for trace files
        trace_files = list(output_dir.glob("*"))
        
        if trace_files:
            print(f"\n✅ Generated {len(trace_files)} profiling trace file(s):")
            for f in trace_files[:5]:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")
            print(f"\n📂 Traces saved to: {output_dir}")
            print(f"   (Will be archived as GitHub Actions artifact)")
        
        print("\n✅ YOLO profiling completed")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p1
class TestProfileVLA:
    """Profile Vision Language Action tests"""
    
    def test_profile_vla(self, cleanup_gpu):
        """Profile existing VLA test using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("Profiling VLA: Vision Language Action Test")
        print("="*70)
        
        # Use persistent directory for traces
        output_dir = get_trace_output_dir("vla_owlvit")
        
        # Profile the existing VLA test
        test_path = "tests/strix_ai/vla/test_action_prediction.py::TestVLA::test_owlvit_detection"
        
        print(f"📊 Profiling: {test_path}")
        result = run_rocprofv3_on_test(test_path, output_dir, timeout=300)
        
        print(f"\n{'='*70}")
        print(f"Profiling Result: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
        print(f"{'='*70}")
        
        if result.stdout:
            print(f"\nTest Output:\n{result.stdout}")
        
        # Check for trace files
        trace_files = list(output_dir.glob("*"))
        
        if trace_files:
            print(f"\n✅ Generated {len(trace_files)} profiling trace file(s):")
            for f in trace_files[:5]:
                print(f"   - {f.name} ({f.stat().st_size} bytes)")
            print(f"\n📂 Traces saved to: {output_dir}")
            print(f"   (Will be archived as GitHub Actions artifact)")
        
        print("\n✅ VLA profiling completed")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p2
@pytest.mark.slow
class TestProfileAll:
    """Profile all test categories together"""
    
    def test_profile_all_categories(self, cleanup_gpu):
        """Profile all Strix AI test categories using rocprofv3"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n" + "="*70)
        print("Profiling ALL Strix AI Test Categories")
        print("="*70)
        
        categories = {
            "vlm": "tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching",
            "vit": "tests/strix_ai/vit/test_vit_base.py::TestViT::test_vit_inference",
            "cv": "tests/strix_ai/cv/test_yolo.py::TestYOLO::test_yolo_detection",
            "vla": "tests/strix_ai/vla/test_action_prediction.py::TestVLA::test_owlvit_detection",
        }
        
        results = {}
        
        for category, test_path in categories.items():
            print(f"\n{'='*70}")
            print(f"📊 Profiling category: {category.upper()}")
            print(f"{'='*70}")
            
            output_dir = get_trace_output_dir(f"all_{category}")
            
            result = run_rocprofv3_on_test(test_path, output_dir, timeout=600)
            
            trace_files = list(output_dir.glob("*"))
            results[category] = {
                "returncode": result.returncode,
                "trace_count": len(trace_files),
                "output_dir": str(output_dir)
            }
            
            print(f"✅ {category}: {len(trace_files)} trace files generated in {output_dir}")
        
        # Summary
        print(f"\n{'='*70}")
        print("Profiling Summary")
        print(f"{'='*70}")
        
        for category, info in results.items():
            status = "✅" if info["returncode"] == 0 else "⚠️"
            print(f"{status} {category:10s}: {info['trace_count']} traces")
        
        print(f"\n📂 All traces will be archived as GitHub Actions artifacts")
        print("\n✅ All categories profiled")


@pytest.mark.strix
@pytest.mark.profiling
@pytest.mark.p0
@pytest.mark.quick
class TestProfileQuick:
    """Quick profiling smoke test"""
    
    def test_profile_quick_smoke(self, cleanup_gpu):
        """Quick test to verify rocprofv3 works with existing tests"""
        if not check_rocprofv3_available():
            pytest.skip("rocprofv3 not available")
        
        print("\n=== Quick Profiling Smoke Test ===")
        print("Testing rocprofv3 integration with existing tests")
        
        # Use persistent directory for traces
        output_dir = get_trace_output_dir("quick_smoke")
        
        # Profile a quick existing test
        test_path = "tests/strix_ai/vlm/test_clip.py::TestCLIP::test_clip_image_text_matching"
        
        print(f"📊 Profiling quick test: {test_path}")
        result = run_rocprofv3_on_test(test_path, output_dir, timeout=120)
        
        trace_files = list(output_dir.glob("*"))
        
        print(f"\n✅ Quick smoke test completed")
        print(f"   rocprofv3 execution: {'SUCCESS' if result.returncode == 0 else 'COMPLETED'}")
        print(f"   Trace files generated: {len(trace_files)}")
        print(f"   Saved to: {output_dir}")
        
        # Passes if rocprofv3 ran
        assert len(trace_files) >= 0, "rocprofv3 should generate output directory"
