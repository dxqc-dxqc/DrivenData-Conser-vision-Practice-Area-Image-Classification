param(
    [ValidateRange(1, 10000)] [int] $Epochs = 15,
    [ValidateRange(1, 1024)] [int] $BatchSize = 8,
    [ValidateRange(0, 10000)] [int] $BenchmarkImages = 0
)

$python = Join-Path $PSScriptRoot '.venv-cuda/Scripts/python.exe'
$trainer = Join-Path $PSScriptRoot 'train_speciesnet.py'
$split = Join-Path $PSScriptRoot 'split_runs/run_20261120_203324_14d448'
$trainingArgs = @('--split', $split, '--batch-size', $BatchSize, '--epochs', $Epochs)
if ($BenchmarkImages -gt 0) {
    $trainingArgs += @('--benchmark', $BenchmarkImages)
}

& $python -B $trainer @trainingArgs
exit $LASTEXITCODE
