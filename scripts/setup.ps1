# First-time setup of agent.cpp on Windows. `agent setup` downloads and runs this file from the repository (scripts/setup.ps1), and the
# installer runs it at the end, so setup can be improved without a new release. It only uses commands the installed `agent` has.
#
#   agent setup [--yes] [--runtimes | --no-runtimes] [--model LINK | --no-model] [--home DIR]
#
# What it does: asks where agent.cpp keeps its files (when this agent can move it), offers the optional private runtimes (bun and uv, for MCP
# servers and skills), and downloads a model from a Hugging Face link or file. The GPU backend and the engines are not asked: they come with
# the build that was installed. With --yes, or without a console, nothing is asked.
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest)
$ErrorActionPreference = 'Stop'

function Step($text) { Write-Host '  ' -NoNewline; Write-Host ([string][char]0x2713) -ForegroundColor Cyan -NoNewline; Write-Host " $text" }
function Info($text) { Write-Host "  $([char]0x00B7) $text" -ForegroundColor DarkGray }
function Fail($text) { Write-Host ''; Write-Host "  $([char]0x2717) $text" -ForegroundColor Magenta; Write-Host ''; exit 1 }

$yes = $false; $runtimes = ''; $model = ''; $noModel = $false; $home_ = ''
for ($i = 0; $i -lt $Rest.Count; $i++) {
    switch ($Rest[$i]) {
        { $_ -in '--yes', '-y' } { $yes = $true }
        { $_ -in '--runtimes', '--install-runtimes' } { $runtimes = 'y' }
        '--no-runtimes' { $runtimes = 'n' }
        '--model' { $i++; $model = $Rest[$i] }
        { $_ -in '--no-model', '--no-models' } { $noModel = $true }
        '--home' { $i++; $home_ = $Rest[$i] }
        '--reconfigure' { }
        { $_ -in '--backend', '--caps', '--js-runtime' } { $i++ } # accepted for the old command line, ignored
        default { Fail "unknown option $($Rest[$i])" }
    }
}

$agent = $env:AGENT_BIN
if (-not $agent) { $cmd = Get-Command agent -ErrorAction SilentlyContinue; if ($cmd) { $agent = $cmd.Source } }
if (-not $agent) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Programs\agent-cpp\agent.exe'
    if (Test-Path $fallback) { $agent = $fallback }
}
if (-not $agent) { Fail 'agent is not installed. Install it first: irm https://agent.qozu.fyi/install.ps1 | iex' }

$ask = (-not $yes) -and [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)

Write-Host ''
Write-Host '  agent.cpp' -ForegroundColor Cyan -NoNewline; Write-Host '  setup' -ForegroundColor DarkGray
Write-Host "  $([string]([char]0x2500) * 28)" -ForegroundColor DarkGray
Write-Host ''
Step ((& $agent version 2>$null | Select-Object -First 1))

# 1. Where the files live. Only an agent that knows `agent home` can move it.
& $agent home *> $null
if ($LASTEXITCODE -eq 0) {
    $current = (& $agent home 2>$null | Select-Object -First 1)
    if ($home_) { & $agent home $home_ *> $null; Step "Files go to $home_" }
    elseif ($ask) {
        $answer = Read-Host "  Where should agent.cpp keep its files (models, engines, skills, memory)? [$current]"
        if ($answer -and $answer -ne $current) { & $agent home $answer *> $null; Step "Files go to $answer" } else { Step "Files go to $current" }
    } else { Step "Files go to $current" }
}

# 2. The optional private runtimes.
if (-not $runtimes -and $ask) {
    $answer = Read-Host '  Download the optional bun and uv runtimes, for MCP servers and skills? [y/N]'
    $runtimes = if ($answer -match '^(y|yes)$') { 'y' } else { 'n' }
}
if ($runtimes -eq 'y') {
    foreach ($r in 'bun', 'uv') {
        Write-Host "  $([char]0x2193) Downloading $r" -ForegroundColor Cyan
        & $agent runtime install $r *> $null
        if ($LASTEXITCODE -eq 0) { Step "$r installed" } else { Info "$r could not be installed (agent runtime install $r)" }
    }
} else {
    Info 'Runtimes skipped: agent runtime install bun | uv adds them any time'
}

# 3. A model: a link or a file, not a list.
if (-not $model -and -not $noModel -and $ask) {
    $model = Read-Host '  Model to download now: a Hugging Face link, owner/repo, or a file on this computer (Enter to skip)'
}
# A path dragged into the terminal or pasted from Explorer comes with quotes (and PowerShell adds "& "): they are not part of the name.
if ($model) { $model = ($model.Trim() -replace '^&\s+', '').Trim().Trim('"').Trim("'").Trim() }
if ($model) {
    Write-Host "  $([char]0x2193) Adding $model" -ForegroundColor Cyan
    & $agent model pull $model
    if ($LASTEXITCODE -ne 0) { Fail 'The model could not be added.' }
}

$listed = (& $agent model ls 2>$null | Out-String).Trim()
$haveModel = $listed -and -not $listed.StartsWith('no models')
if ($haveModel -and $ask) {
    Write-Host ''
    Write-Host '  Done. Starting the chat ...'
    Write-Host ''
    & $agent
} elseif ($haveModel) {
    Write-Host ''
    Write-Host '  Done. Run ' -NoNewline; Write-Host 'agent' -ForegroundColor Cyan -NoNewline; Write-Host ' to start the chat.'
    Write-Host ''
} else {
    Write-Host ''
    Write-Host '  Done. Add a model with ' -NoNewline; Write-Host 'agent model pull <Hugging Face link or file>' -ForegroundColor Cyan -NoNewline; Write-Host ', then run ' -NoNewline; Write-Host 'agent' -ForegroundColor Cyan -NoNewline; Write-Host '.'
    Write-Host ''
}
