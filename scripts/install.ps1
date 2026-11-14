# Installs the agent CLI on Windows:  irm https://agent.qozu.fyi | iex
#
# Picks the release archive that fits this machine (x64 or arm64; the Vulkan build on x64 when a Vulkan driver is present), checks its
# SHA-256 against the release's SHA256SUMS, unpacks it to %LOCALAPPDATA%\Programs\agent-cpp and adds that folder to your user PATH.
# Environment: AGENT_VERSION (a tag such as v0.2.0; default: the latest release), AGENT_VARIANT (cpu or vulkan; default: detected),
#              AGENT_REPO (default amitverma-cf/agent.cpp).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repo = if ($env:AGENT_REPO) { $env:AGENT_REPO } else { 'amitverma-cf/agent.cpp' }
$arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    'AMD64' { 'x86_64' }
    'ARM64' { 'aarch64' }
    default { throw "unsupported CPU: $($env:PROCESSOR_ARCHITECTURE) (x64 and arm64 are released)" }
}
$variant = $env:AGENT_VARIANT
if (-not $variant) {
    $variant = 'cpu'
    if ($arch -eq 'x86_64' -and (Test-Path "$env:SystemRoot\System32\vulkan-1.dll")) { $variant = 'vulkan' }
}
if ($variant -ne 'cpu' -and $variant -ne 'vulkan') { throw 'AGENT_VARIANT must be cpu or vulkan' }

$base = if ($env:AGENT_VERSION) { "https://github.com/$repo/releases/download/$($env:AGENT_VERSION)" } else { "https://github.com/$repo/releases/latest/download" }
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("agent-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    # GitHub serves SHA256SUMS as application/octet-stream: Content is then bytes, not text.
    $resp = Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS"
    $text = if ($resp.Content -is [byte[]]) { [System.Text.Encoding]::UTF8.GetString($resp.Content) } else { [string]$resp.Content }
    $sums = $text -split "`r?`n" | Where-Object { $_.Trim() }
    function Pick($suffix) {
        $sums | Where-Object { $_ -match " \*?agent-cpp-\S*-windows-$arch$suffix\.zip\s*$" } | Select-Object -First 1
    }
    $line = if ($variant -eq 'vulkan') { Pick '-vulkan' } else { Pick '' }
    if (-not $line -and $variant -eq 'vulkan') { Write-Host 'no Vulkan build in this release; using the CPU build'; $line = Pick '' }
    if (-not $line) { throw "this release has no Windows $arch archive" }
    $parts = $line.Trim() -split '\s+', 2
    $want = $parts[0].ToLower()
    $name = $parts[1].TrimStart('*')

    Write-Host "downloading $name"
    $zip = Join-Path $tmp $name
    Invoke-WebRequest -UseBasicParsing "$base/$name" -OutFile $zip
    if ((Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower() -ne $want) { throw "SHA-256 mismatch for $name; nothing was installed" }

    $dest = Join-Path $env:LOCALAPPDATA 'Programs\agent-cpp'
    $stage = Join-Path $tmp 'stage'
    Expand-Archive -Path $zip -DestinationPath $stage
    $inner = Get-ChildItem $stage | Select-Object -First 1
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
    Move-Item $inner.FullName $dest

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $dest) {
        [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $dest), 'User')
        Write-Host "added $dest to your user PATH (open a new terminal)"
    }
    Write-Host "installed $name to $dest"
    Write-Host 'next: agent setup'
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
