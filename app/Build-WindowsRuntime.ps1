# Run from an Ilaria RVC installation with the updated launcher.
# Requires uv and a compatible NVIDIA driver.
param([ValidateSet("cu128","cu126")][string]$Backend = "cu128")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$runtime = Join-Path $root "runtime-py310"
$python = Join-Path $runtime "Scripts/python.exe"
$requirements = Join-Path $root "requirements-windows-py310.txt"
$marker = Join-Path $runtime ".cuda-verified"

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE." }
}

if (Test-Path -LiteralPath $marker) {
    throw "The upgraded runtime is already verified. Choose a fresh installation folder to rebuild it."
}
if (!(Test-Path -LiteralPath $python)) {
    Invoke-Checked uv @("venv","--managed-python","--python","3.10","--seed",$runtime)
}
Invoke-Checked uv @("pip","install","--python",$python,"--index","https://download.pytorch.org/whl/$Backend",
    "torch==2.10.0+$Backend","torchaudio==2.10.0+$Backend","torchvision==0.25.0+$Backend")
Invoke-Checked uv @("pip","install","--python",$python,"-r",$requirements,
    "pip==24.0","setuptools==69.5.1","wheel==0.43.0","Cython==0.29.37")
Invoke-Checked $python @("-m","pip","install","--no-build-isolation","fairseq==0.12.2","hydra-core==1.0.7","omegaconf==2.0.6")
Invoke-Checked $python @("-m","pip","check")
$smoke = @"
import sys
from pathlib import Path
import torch
import torchaudio, torchvision, gradio, faiss, pyworld, av
from ilaria_embedders import load_fairseq_hubert
assert torch.__version__ == '2.10.0+$Backend'
assert torch.cuda.is_available(), 'CUDA is unavailable. Check the NVIDIA driver.'
for gpu in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{gpu}')
    x = torch.ones((16, 16), device=device, requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize(device)
checkpoint = Path('assets/hubert/hubert_base.pt')
models, _, _ = load_fairseq_hubert(checkpoint)
with torch.inference_mode():
    model = models[0].eval().cuda()
    result = model.extract_features(source=torch.zeros((1, 16000), device='cuda'), output_layer=9)
    assert result[0].shape[-1] == 768
"@
Push-Location $root
try {
    Invoke-Checked $python @("-c", "import sys; sys.path.insert(0, 'ilaria'); " + $smoke)
    Set-Content -LiteralPath $marker -Value "2.10.0+$Backend" -Encoding ascii
} finally {
    Pop-Location
}
Write-Host "Verified PyTorch 2.10.0+$Backend. Start Ilaria RVC with ILARIA_RVC.bat."
