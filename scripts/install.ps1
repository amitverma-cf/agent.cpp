# Installs the agent CLI on Windows:  irm https://agent.qozu.fyi/install.ps1 | iex
#
# Picks the release archive that fits this machine (x64 or arm64; the Vulkan build on x64 when a Vulkan driver is present), checks its
# SHA-256 against the release's SHA256SUMS, unpacks it to %LOCALAPPDATA%\Programs\agent-cpp and adds that folder to your user PATH.
# It then runs the setup script (scripts/setup.ps1) and opens the chat (AGENT_NO_START=1 only installs). Run again, it updates: the same
# version is left alone, a newer one replaces it in place (`agent update` does exactly this).
# Environment: AGENT_VERSION (a tag such as v1.0.0; default: the latest release), AGENT_VARIANT (cpu or vulkan; default: detected),
#              AGENT_REPO (default amitverma-cf/agent.cpp).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# `agent update` starts this script and exits: Windows will not replace a program that is still running, so wait for it.
if ($env:AGENT_WAIT_PID) { Wait-Process -Id ([int]$env:AGENT_WAIT_PID) -Timeout 30 -ErrorAction SilentlyContinue }

# ---- output: turquoise accents, grey detail, pink for problems
function Step($text) { Write-Host '  ' -NoNewline; Write-Host ([string][char]0x2713) -ForegroundColor Cyan -NoNewline; Write-Host " $text" }
function Info($text) { Write-Host "  $([char]0x00B7) $text" -ForegroundColor DarkGray }
function Fail($text) { Write-Host ''; Write-Host "  $([char]0x2717) $text" -ForegroundColor Magenta; Write-Host ''; throw $text }

Write-Host ''
Write-Host '  agent.cpp' -ForegroundColor Cyan -NoNewline; Write-Host '  installer' -ForegroundColor DarkGray
Write-Host "  $([string]([char]0x2500) * 28)" -ForegroundColor DarkGray
Write-Host ''

$tmp = $null
try {
    $repo = if ($env:AGENT_REPO) { $env:AGENT_REPO } else { 'amitverma-cf/agent.cpp' }
    $arch = switch ($env:PROCESSOR_ARCHITECTURE) {
        'AMD64' { 'x86_64' }
        'ARM64' { 'aarch64' }
        default { Fail "Unsupported CPU: $($env:PROCESSOR_ARCHITECTURE). x64 and arm64 are released." }
    }
    $variant = $env:AGENT_VARIANT
    $detected = ''
    if (-not $variant) {
        $variant = 'cpu'
        if ($arch -eq 'x86_64' -and (Test-Path "$env:SystemRoot\System32\vulkan-1.dll")) { $variant = 'vulkan'; $detected = ' (Vulkan driver found)' }
    }
    if ($variant -notin 'cpu', 'vulkan') { Fail 'AGENT_VARIANT must be cpu or vulkan.' }

    $base = if ($env:AGENT_VERSION) { "https://github.com/$repo/releases/download/$($env:AGENT_VERSION)" } else { "https://github.com/$repo/releases/latest/download" }
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("agent-install-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null

    # GitHub serves SHA256SUMS as application/octet-stream: Content is then bytes, not text.
    $resp = Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS"
    $text = if ($resp.Content -is [byte[]]) { [System.Text.Encoding]::UTF8.GetString($resp.Content) } else { [string]$resp.Content }
    $sums = $text -split "`r?`n" | Where-Object { $_.Trim() }
    function Pick($suffix) {
        $sums | Where-Object { $_ -match " \*?agent-cpp-\S*-windows-$arch$suffix\.zip\s*$" } | Select-Object -First 1
    }
    $line = if ($variant -eq 'vulkan') { Pick '-vulkan' } else { Pick '' }
    if (-not $line -and $variant -eq 'vulkan') { Info 'No GPU build in this release; using the CPU build'; $variant = 'cpu'; $detected = ''; $line = Pick '' }
    if (-not $line) { Fail "This release has no Windows $arch archive." }
    $parts = $line.Trim() -split '\s+', 2
    $want = $parts[0].ToLower()
    $name = $parts[1].TrimStart('*')
    $version = if ($name -match '^agent-cpp-(v?\d[^-]*)-') { $Matches[1] } else { '' }

    Step "Windows $arch  $([char]0x00B7)  $variant build$detected"
    Step "Release $version"

    $dest = Join-Path $env:LOCALAPPDATA 'Programs\agent-cpp'
    $versionFile = Join-Path $dest '.version'
    if ($env:AGENT_CHECK) { # agent update --check: only look
        if ((Test-Path $versionFile) -and ((Get-Content $versionFile -Raw).Trim() -eq $name)) { Step "Up to date ($version)" } else { Step "An update is available: $version" }
        return
    }
    $upToDate = (-not $env:AGENT_FORCE) -and (Test-Path $versionFile) -and ((Get-Content $versionFile -Raw).Trim() -eq $name) -and (Test-Path (Join-Path $dest 'agent.exe'))
    if ($upToDate) { Step 'Already installed and up to date' }
    else {

    # Download with a progress bar.
    Write-Host "  $([char]0x2193) Downloading $name" -ForegroundColor Cyan
    $zip = Join-Path $tmp $name
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromMinutes(30)
    $r = $client.GetAsync("$base/$name", [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
    [void]$r.EnsureSuccessStatusCode()
    $total = $r.Content.Headers.ContentLength
    $in = $r.Content.ReadAsStreamAsync().Result
    $out = [System.IO.File]::Create($zip)
    $buf = New-Object byte[] 262144
    $done = 0; $shown = -1
    while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
        $out.Write($buf, 0, $n); $done += $n
        if ($total) {
            $pct = [int](100 * $done / $total)
            if ($pct -ne $shown) {
                $shown = $pct
                $filled = [int]($pct / 4)
                Write-Host ("`r    [" + ('#' * $filled) + (' ' * (25 - $filled)) + "] $pct%  " + [math]::Round($done / 1MB, 1) + ' / ' + [math]::Round($total / 1MB, 1) + ' MB') -NoNewline -ForegroundColor DarkGray
            }
        }
    }
    $out.Dispose(); $in.Dispose(); $client.Dispose()
    Write-Host ''
    if ((Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower() -ne $want) { Fail "SHA-256 mismatch for $name. Nothing was installed." }
    Step 'Checksum verified (SHA-256)'

    $dest = Join-Path $env:LOCALAPPDATA 'Programs\agent-cpp'
    $stage = Join-Path $tmp 'stage'
    Expand-Archive -Path $zip -DestinationPath $stage
    $inner = Get-ChildItem $stage | Select-Object -First 1
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Move-Item $inner.FullName $dest
    Step "Installed to $dest"
    Set-Content -Path $versionFile -Value $name -NoNewline

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $dest) {
        [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $dest), 'User')
        Step 'Added to your PATH (open a new terminal to type just agent)'
    }
    }
} finally {
    if ($tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
}

# Setup is its own script (scripts/setup.ps1), fetched from the repository so it can be improved without a new release. `agent update` sets
# AGENT_NO_START: an update does not set anything up again. The PATH change only reaches new terminals, so the full path is used here.
$env:Path += ";$dest"
$exe = Join-Path $dest 'agent.exe'
$scripts = if ($env:AGENT_SCRIPTS) { $env:AGENT_SCRIPTS } else { 'https://raw.githubusercontent.com/amitverma-cf/agent.cpp/main/scripts' }
if (-not $env:AGENT_NO_START -and [Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
    $setupText = $null
    try { $setupText = (Invoke-WebRequest -UseBasicParsing "$scripts/setup.ps1").Content } catch { }
    if ($setupText -is [byte[]]) { $setupText = [System.Text.Encoding]::UTF8.GetString($setupText) }
    if ($setupText) {
        $env:AGENT_BIN = $exe
        & ([scriptblock]::Create($setupText))
    } else {
        Info 'Could not fetch the setup script; run agent setup later'
    }
} else {
    if (-not $upToDate) {
        Write-Host ''
        Write-Host '  Next: ' -NoNewline; Write-Host 'agent setup' -ForegroundColor Cyan
        Write-Host ''
    }
}
