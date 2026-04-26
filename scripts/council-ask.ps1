param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Question,

    [switch]$Lite,
    [int]$Rounds = 2,
    [string]$Mode = "",
    [string]$Synthesizer = ""
)

$councilDir = "C:\Users\1028120\Documents\Dev\ai-council"
$councilExe = "$councilDir\.venv\Scripts\council.exe"

# Fall back to globally installed council if no venv present
if (-not (Test-Path $councilExe)) {
    $councilExe = (Get-Command council -ErrorAction SilentlyContinue)?.Source
}
if (-not $councilExe) {
    Write-Error "council not found. Run 'py -m pip install -e .' from $councilDir"
    exit 1
}

$flags = @()
if ($Lite)           { $flags += "--lite" }
if ($Rounds -ne 2)   { $flags += "--rounds"; $flags += $Rounds }
if ($Mode)           { $flags += "-M"; $flags += $Mode }
if ($Synthesizer)    { $flags += "--synthesizer"; $flags += $Synthesizer }

if (Test-Path $Question) {
    & $councilExe --file $Question @flags
} else {
    & $councilExe $Question @flags
}
