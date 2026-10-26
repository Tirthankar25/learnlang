"""Checks whether PyTorch and onnxruntime can both use cuDNN on the GPU.

Run inside the LeLa venv, on the GPU LeLa uses:
    CUDA_VISIBLE_DEVICES=0 python check_gpu.py
and send the whole output.
"""

import glob
import importlib.metadata as md
import os
import subprocess
import sys
import tempfile


def pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "-"


def loaded_cudnn():
    """Which libcudnn files this process actually loaded (system copy vs pip copy)."""
    try:
        with open("/proc/self/maps") as f:
            return sorted({line.split()[-1] for line in f if "libcudnn" in line})
    except OSError:
        return []


print("python      ", sys.version.split()[0])
print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES", "(unset)"))
print("LD_LIBRARY_PATH      =", os.getenv("LD_LIBRARY_PATH", "(unset)"))
for name in ("torch", "torchaudio", "onnxruntime", "onnxruntime-gpu", "onnx", "ctranslate2",
             "faster-whisper", "transformers", "nvidia-cudnn-cu12", "nvidia-cudnn-cu11",
             "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
    print(f"  {name:26s} {pkg(name)}")
if pkg("onnxruntime") != "-" and pkg("onnxruntime-gpu") != "-":
    print("  !! both onnxruntime and onnxruntime-gpu are installed; they overwrite each other."
          " Fix: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.20.2")

try:
    out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    system = sorted({l.split("=>")[-1].strip() for l in out.splitlines() if "libcudnn.so" in l})
    print("system cuDNN (ldconfig):", system or "none")
except FileNotFoundError:
    pass

print("\n--- PyTorch")
import torch

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda,
      "| cuDNN", torch.backends.cudnn.version(), "| GPUs visible", torch.cuda.device_count())
torch_ok = False
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.enabled = True
        x = torch.randn(1, 4, 256, device="cuda")
        torch.nn.Conv1d(4, 4, 3).cuda()(x)
        torch.cuda.synchronize()
        torch_ok = True
        print("PyTorch cuDNN conv: OK")
    except Exception as e:
        print("PyTorch cuDNN conv: FAILED ->", str(e).splitlines()[0])
print("libcudnn loaded after torch:", loaded_cudnn() or "none")

print("\n--- onnxruntime")
try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    print("onnxruntime", ort.__version__, "| providers", ort.get_available_providers())
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 4, 3], [0.1] * 48)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 256])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = os.path.join(tempfile.mkdtemp(), "conv.onnx")
    onnx.save(model, path)
    try:
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        import numpy as np

        sess.run(None, {"x": np.random.randn(1, 4, 256).astype("float32")})
        print("onnxruntime CUDA conv:", "OK" if sess.get_providers()[0] == "CUDAExecutionProvider"
              else f"fell back to {sess.get_providers()}")
    except Exception as e:
        print("onnxruntime CUDA conv: FAILED ->", str(e).splitlines()[0][:300])
except ImportError as e:
    print("onnxruntime/onnx not importable:", e)
print("libcudnn loaded after onnxruntime:", loaded_cudnn() or "none")

pip_cudnn = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cudnn", "lib", "libcudnn.so*"))
print("\npip cuDNN in this venv:", pip_cudnn or "none")
