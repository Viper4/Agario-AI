param(
    [int]$mbras = 2,
    [int]$rnns = 2,
    [int]$numRuns = 10,
    [string]$output = "results.csv",
    [string]$snapshots = "agent_snapshots.pkl",
    [int]$duration = 240
)

$py = "python"
$args = "run_headless.py --mbras $mbras --rnns $rnns --num-runs $numRuns --duration $duration --output $output --snapshots $snapshots --headless"
Write-Host "Running: $py $args"
& $py $args
