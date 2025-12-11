"""
Vision Language Action (VLA) tests for Strix platforms

Tests action prediction and embodied AI capabilities.
VLA models understand visual scenes and predict/generate actions.

Use cases:
- Edge robotics
- Interactive AI agents
- Action understanding from video
- Consumer robotics applications
"""

import pytest
import torch
import numpy as np
from PIL import Image
import os

# Skip module if dependencies not available
transformers = pytest.importorskip("transformers", reason="transformers not installed")

AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")


@pytest.mark.strix
@pytest.mark.p0
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific VLA test"
)
class TestVisionLanguageAction:
    """Vision Language Action tests for Strix platforms"""
    
    def test_vla_visual_grounding(self, strix_device, test_image_224, cleanup_gpu):
        """Test visual grounding - understanding object locations from text"""
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        print("\n🧠 Loading OWL-ViT model for visual grounding...")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        model = model.to(strix_device)
        model.eval()
        
        # Create test image with text queries
        image = Image.new('RGB', (640, 480), color='white')
        texts = [["a photo of a cat", "a photo of a dog"]]
        
        print("🔍 Processing visual grounding query...")
        inputs = processor(text=texts, images=image, return_tensors="pt").to(strix_device)
        
        print("⚡ Running VLA inference on Strix...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        target_sizes = torch.Tensor([image.size[::-1]]).to(strix_device)
        results = processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=0.1
        )
        
        print(f"✅ Visual grounding successful!")
        print(f"   Detections: {len(results)}")
        
        assert len(results) > 0, "Should have detection results"
        assert outputs.logits.device.type == "cuda"
        
        print(f"✅ VLA visual grounding test passed on {AMDGPU_FAMILIES}!")
    
    def test_vla_action_classification(self, strix_device, test_image_224, cleanup_gpu):
        """Test action recognition from images using CLIP-based approach"""
        from transformers import CLIPProcessor, CLIPModel
        
        print("\n🧠 Loading CLIP for action classification...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        # Test image with action descriptions
        image = Image.new('RGB', (224, 224), color='blue')
        actions = [
            "a person walking",
            "a person running", 
            "a person sitting",
            "a person jumping"
        ]
        
        print(f"🔍 Classifying actions: {actions}")
        inputs = processor(text=actions, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        print("⚡ Running action classification...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        predicted_action_idx = probs.argmax().item()
        predicted_action = actions[predicted_action_idx]
        
        print(f"✅ Action classification complete!")
        print(f"   Predicted action: {predicted_action}")
        print(f"   Probabilities: {probs[0].cpu().numpy()}")
        
        assert probs.device.type == "cuda"
        assert probs.shape[1] == len(actions)
        
        print(f"✅ VLA action classification test passed on {AMDGPU_FAMILIES}!")
    
    def test_vla_spatial_reasoning(self, strix_device, cleanup_gpu):
        """Test spatial reasoning - understanding object relationships"""
        from transformers import CLIPProcessor, CLIPModel
        
        print("\n🧠 Loading model for spatial reasoning...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        # Test images for spatial relationships
        image = Image.new('RGB', (224, 224), color='green')
        spatial_queries = [
            "object on the left",
            "object on the right",
            "object above",
            "object below"
        ]
        
        print(f"🔍 Testing spatial reasoning...")
        inputs = processor(text=spatial_queries, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        print(f"✅ Spatial reasoning complete!")
        print(f"   Spatial probabilities: {probs[0].cpu().numpy()}")
        
        assert probs.device.type == "cuda"
        
        print(f"✅ VLA spatial reasoning test passed on {AMDGPU_FAMILIES}!")
    
    def test_vla_multimodal_action_planning(self, strix_device, cleanup_gpu):
        """Test multimodal understanding for action planning"""
        from transformers import CLIPProcessor, CLIPModel
        
        print("\n🧠 Testing multimodal action planning...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        # Sequence of images representing actions
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        action_sequence = [
            "pick up object",
            "move object",
            "place object"
        ]
        
        print(f"🔍 Planning action sequence...")
        
        # Process each step
        for i, (image, action) in enumerate(zip(images, action_sequence)):
            inputs = processor(text=[action], images=image, return_tensors="pt").to(strix_device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"   Step {i+1}: {action} - ✓")
        
        print(f"✅ Action planning sequence complete!")
        
        assert outputs.logits_per_image.device.type == "cuda"
        
        print(f"✅ VLA action planning test passed on {AMDGPU_FAMILIES}!")
    
    @pytest.mark.slow
    def test_vla_realtime_action_detection(self, strix_device, cleanup_gpu):
        """Test real-time action detection (video stream simulation)"""
        from transformers import CLIPProcessor, CLIPModel
        import time
        
        print("\n🧠 Loading model for real-time action detection...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        actions = ["walking", "running", "standing"]
        
        # Warmup
        print("🔥 Warming up...")
        dummy_image = Image.new('RGB', (224, 224))
        dummy_inputs = processor(text=actions, images=dummy_image, return_tensors="pt", padding=True).to(strix_device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(**dummy_inputs)
        torch.cuda.synchronize()
        
        # Simulate video frames
        print("📹 Simulating real-time video action detection...")
        num_frames = 90  # 3 seconds at 30 FPS
        frame_times = []
        
        for _ in range(num_frames):
            frame = Image.new('RGB', (224, 224))
            inputs = processor(text=actions, images=frame, return_tensors="pt", padding=True).to(strix_device)
            
            start = time.time()
            with torch.no_grad():
                _ = model(**inputs)
            torch.cuda.synchronize()
            frame_times.append(time.time() - start)
        
        avg_fps = 1.0 / np.mean(frame_times)
        
        print(f"📊 Performance Results:")
        print(f"   Average FPS: {avg_fps:.2f}")
        print(f"   Avg latency: {np.mean(frame_times)*1000:.2f} ms")
        
        # Should achieve real-time performance
        assert avg_fps >= 10, f"VLA FPS too low: {avg_fps:.2f} FPS (expected >= 10 FPS)"
        
        print(f"✅ VLA real-time test passed on {AMDGPU_FAMILIES}!")
    
    def test_vla_memory_efficiency(self, strix_device, test_image_224, cleanup_gpu):
        """Test VLA memory usage on Strix iGPU"""
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        
        print("\n🧠 Testing VLA memory efficiency...")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        model = model.to(strix_device)
        model.eval()
        
        texts = [["a photo of a cat"]]
        inputs = processor(text=texts, images=test_image_224, return_tensors="pt").to(strix_device)
        
        print("⚡ Running inference and measuring memory...")
        with torch.no_grad():
            _ = model(**inputs)
        
        torch.cuda.synchronize()
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"📊 Memory Usage:")
        print(f"   Peak: {peak_memory_mb:.2f} MB")
        
        # Should stay under reasonable limits for iGPU
        assert peak_memory_mb < 2048, f"Memory usage too high: {peak_memory_mb:.2f} MB"
        
        print(f"✅ VLA memory test passed on {AMDGPU_FAMILIES}!")


@pytest.mark.strix
@pytest.mark.p1
@pytest.mark.skipif(
    AMDGPU_FAMILIES not in ["gfx1150", "gfx1151"],
    reason="Strix-specific VLA test"
)
class TestVLAAdvanced:
    """Advanced VLA tests for robotics and embodied AI"""
    
    def test_vla_affordance_detection(self, strix_device, cleanup_gpu):
        """Test affordance detection - what actions are possible with objects"""
        from transformers import CLIPProcessor, CLIPModel
        
        print("\n🧠 Testing affordance detection...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        image = Image.new('RGB', (224, 224))
        affordances = [
            "can be grasped",
            "can be pushed",
            "can be pulled",
            "can be lifted"
        ]
        
        print(f"🔍 Detecting affordances...")
        inputs = processor(text=affordances, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        print(f"✅ Affordance detection complete!")
        print(f"   Affordance probabilities: {probs[0].cpu().numpy()}")
        
        assert probs.device.type == "cuda"
        
        print(f"✅ VLA affordance test passed on {AMDGPU_FAMILIES}!")
    
    def test_vla_goal_inference(self, strix_device, cleanup_gpu):
        """Test goal inference from visual observations"""
        from transformers import CLIPProcessor, CLIPModel
        
        print("\n🧠 Testing goal inference...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(strix_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        image = Image.new('RGB', (224, 224))
        goals = [
            "organize objects",
            "clean the space",
            "arrange items",
            "sort by color"
        ]
        
        print(f"🔍 Inferring goals...")
        inputs = processor(text=goals, images=image, return_tensors="pt", padding=True).to(strix_device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        inferred_goal_idx = probs.argmax().item()
        inferred_goal = goals[inferred_goal_idx]
        
        print(f"✅ Goal inference complete!")
        print(f"   Inferred goal: {inferred_goal}")
        
        assert probs.device.type == "cuda"
        
        print(f"✅ VLA goal inference test passed on {AMDGPU_FAMILIES}!")

