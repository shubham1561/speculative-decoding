<#
===============================================================================
 run_overnight.ps1 -- data, training, distillation, benchmarks. Then go to sleep.
===============================================================================

     cd "<this folder>"        (the python folder of the repo)
     powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1

 Everything is written to logs\overnight_<date>_<time>.log as well as the
 screen, so nothing is lost if the window closes.

 THE ORDER (rough times on a 6 GB GPU)
 -------------------------------------
   1  prepare_data.py                     TinyStories -> data\   (10-20 min)
   2  train_scratch.py --model target     checkpoints\target.pt  (1-2 h)
   3  train_scratch.py --model draft      checkpoints\draft.pt   (20-40 min)
   4  distill_draft.py --pair scratch     draft_distilled.pt     (20-40 min)
   5  benchmark.py --pair scratch         results\               (5-15 min)
   6  distill_draft.py --pair gpt2        downloads gpt2-large (3.2 GB) and
                                          distilgpt2 once; trains a better
                                          draft for it            (45-90 min)
   7  benchmark.py --pair gpt2            stories + general prompts (30-60 min)
   8  benchmark.py --pair qwen            only with -WithQwen (downloads ~4 GB)

 -OnlyGpt2 runs steps 6-7 only (run_gpt2.bat does that with a double-click).

 Every step can be re-run safely. Training steps use --resume, so if the PC
 restarts in the night, run the script again: finished steps end at once and
 an unfinished one continues from its last checkpoint. Data is not
 downloaded again if data\tinystories\train.bin exists.

 A failure in a benchmark step does not stop the rest. A failure in a
 training step stops the steps that need its model.

 The computer is kept awake for the duration (the screen may still sleep).
===============================================================================
#>

param(
    [switch]$SkipGpt2,         # skip steps 6-7 (no big downloads)
    [switch]$OnlyGpt2,         # only steps 6-7 (your own models are already trained)
    [switch]$WithQwen,         # also run step 8
    [int]$MaxStories = 0       # step 1: 0 = all of TinyStories; e.g. 300000 for a quicker run
)

$ErrorActionPreference = 'Continue'
$PY = $PSScriptRoot
$PYEXE = Join-Path $PY ".venv\Scripts\python.exe"
if (-not (Test-Path $PYEXE)) { $PYEXE = "python" }
Set-Location -LiteralPath $PY
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PY "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir "overnight_$stamp.log"

# keep the PC awake while this runs
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$results = [ordered]@{}
function Step([string]$name, [string[]]$pyArgs) {
    $line = "`n===== $name =====   ($(Get-Date -Format 'HH:mm:ss'))"
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line -Encoding UTF8
    $t0 = Get-Date
    & $PYEXE @pyArgs 2>&1 | ForEach-Object {
        $text = "$_"
        Write-Host $text
        Add-Content -Path $log -Value $text -Encoding UTF8
    }
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $script:results[$name] = if ($code -eq 0) { "OK    ($mins min)" } else { "FAILED (exit $code, $mins min)" }
    return ($code -eq 0)
}

try {
    $dataOk = Test-Path (Join-Path $PY "data\tinystories\train.bin")
    if (-not $dataOk) {
        $a = @("prepare_data.py")
        if ($MaxStories -gt 0) { $a += @("--max_stories", "$MaxStories") }
        $dataOk = Step "1 prepare data" $a
    } else { $results["1 prepare data"] = "skipped (data already there)" }

    if ($dataOk) {
        if ($OnlyGpt2) {
            $targetOk = $false
            $results["2-5 own models"] = "skipped (-OnlyGpt2)"
        } else {
            $targetOk = Step "2 train target" @("train_scratch.py", "--model", "target", "--resume")
            $draftOk  = Step "3 train draft"  @("train_scratch.py", "--model", "draft", "--resume")
        }
        if ($targetOk -and $draftOk) {
            Step "4 distill draft (scratch)" @("distill_draft.py", "--pair", "scratch", "--resume") | Out-Null
            Step "5 benchmark scratch" @("benchmark.py", "--pair", "scratch") | Out-Null
        }
        if (-not $SkipGpt2) {
            Step "6 distill draft (gpt2)" @("distill_draft.py", "--pair", "gpt2", "--resume") | Out-Null
            Step "7a benchmark gpt2 (stories)" @("benchmark.py", "--pair", "gpt2", "--prompts", "stories") | Out-Null
            Step "7b benchmark gpt2 (general)" @("benchmark.py", "--pair", "gpt2", "--prompts", "general") | Out-Null
        }
        if ($WithQwen) {
            Step "8 benchmark qwen (general)" @("benchmark.py", "--pair", "qwen", "--prompts", "general") | Out-Null
        }
    }
}
finally {
    [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null   # let the PC sleep again
    $summary = "`n===== SUMMARY =====`n" + (($results.GetEnumerator() | ForEach-Object { "{0,-30} {1}" -f $_.Key, $_.Value }) -join "`n")
    $summary += "`n`nResults table: results\RESULTS.md     Full log: $log"
    Write-Host $summary -ForegroundColor Green
    Add-Content -Path $log -Value $summary -Encoding UTF8
}
