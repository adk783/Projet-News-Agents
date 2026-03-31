"""Check if model was saved successfully."""
import os

model_dir = "./uncertainty_model"
required_files = ["config.json", "model.safetensors", "tokenizer.json"]

print(f"Checking model directory: {model_dir}")
if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    print(f"Files found: {len(files)}")
    for f in sorted(files):
        full = os.path.join(model_dir, f)
        if os.path.isfile(full):
            size = os.path.getsize(full)
            print(f"  {f}: {size:,} bytes")
        else:
            print(f"  {f}/ (directory)")
    
    # Check if key files exist
    for rf in required_files:
        path = os.path.join(model_dir, rf)
        if os.path.exists(path):
            print(f"  ✅ {rf} found")
        else:
            print(f"  ❌ {rf} MISSING")
else:
    print("❌ Model directory not found!")
