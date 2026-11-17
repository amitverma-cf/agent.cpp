# Tests the modular install on Windows: the installer script, with the files of a release build (a folder with SHA256SUMS and the
# agent-cpp-core-* and agent-engine-* archives) and the real upstream downloads, in a folder of its own. Nothing outside the folder is
# touched: no user environment variable, no PATH entry (AGENT_NO_ENV).
#
#   powershell -File run.ps1 -Release C:\path\to\release -Scripts C:\path\to\scripts [-Tag local] [-Engines llama,onnx] [-Runtimes bun,uv,node,pnpm,nub]
param([string]$Release, [string]$Scripts, [string]$Tag = 'local', [string[]]$Engines = @('llama', 'audiocpp', 'sdcpp', 'onnx', 'trellis'),
      [string[]]$Runtimes = @('bun', 'uv', 'node', 'pnpm', 'nub'), [string]$Work = "$env:TEMP\agent-modtest")
$ErrorActionPreference = 'Continue'
$Engines = @($Engines | ForEach-Object { $_ -split ',' } | Where-Object { $_ }); $Runtimes = @($Runtimes | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Path $Work | Out-Null
$root = Join-Path $Work 'root'
$env:AGENT_HOME = $root; $env:AGENT_RELEASE = $Release; $env:AGENT_SCRIPTS = $Scripts; $env:AGENT_VERSION = $Tag
$env:AGENT_NO_ENV = '1'; $env:AGENT_YES = '1'; $env:AGENT_NO_START = '1'; $env:NO_COLOR = '1'
$script:pass = 0; $script:fail = 0
function Check([string]$name, [bool]$ok, [string]$why = '') {
    if ($ok) { $script:pass++; "PASS $name" } else { $script:fail++; "FAIL $name  ::  $why" }
}
$installer = Join-Path $Scripts 'install.ps1'
$userHomeBefore = [Environment]::GetEnvironmentVariable('AGENT_HOME', 'User')
$userPathBefore = [Environment]::GetEnvironmentVariable('Path', 'User')

'== install'
& powershell -NoProfile -ExecutionPolicy Bypass -File $installer install --yes 2>&1 | Select-Object -Last 8
Check 'install: the program is in bin' (Test-Path (Join-Path $root 'bin\agent.exe'))
Check 'install: the marker and the pinned scripts are there' ((Test-Path "$root\.agent-root") -and (Test-Path "$root\scripts\components.tsv") -and (Test-Path "$root\scripts\install.ps1"))
$home_ = (& "$root\bin\agent.exe" home 2>$null | Select-Object -First 1)
Check 'install: the program reports the folder' ($home_ -eq $root) "got '$home_'"
Check 'install: nothing outside the folder' (-not (Get-ChildItem $Work | Where-Object { $_.Name -ne 'root' }))
Check 'install: the user environment is untouched' (([Environment]::GetEnvironmentVariable('AGENT_HOME', 'User') -eq $userHomeBefore) -and ([Environment]::GetEnvironmentVariable('Path', 'User') -eq $userPathBefore))

'== ls'
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" ls 2>&1 | Select-Object -Last 14

'== every engine and runtime'
foreach ($n in ($Engines + $Runtimes)) { & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" add $n 2>&1 | Select-Object -Last 3 }
$doctor = (& "$root\bin\agent.exe" doctor 2>&1 | Out-String)
foreach ($n in $Engines) {
    Check "engine ${n}: installed with its pin" (Test-Path "$root\engines\$n\.pin")
    Check "engine ${n}: the program lists it as ok" ($doctor -match "(?m)^\s+$n\s+ok") ($doctor -split "`n" | Select-String $n | Out-String)
}
foreach ($n in $Runtimes) { Check "runtime ${n}: installed" (Test-Path "$root\runtimes\$n\.pin") }
if ($Runtimes -contains 'bun') { Check 'runtime bun runs' ((& "$root\runtimes\bun\bun.exe" --version 2>&1) -match '^\d') }
if ($Runtimes -contains 'uv') { Check 'runtime uv runs' ((& "$root\runtimes\uv\uv.exe" --version 2>&1) -match 'uv') }
if ($Runtimes -contains 'node') { Check 'runtime node runs' ((& "$root\runtimes\node\node.exe" --version 2>&1) -match '^v\d') }
if ($Runtimes -contains 'pnpm') { Check 'runtime pnpm is there' (Test-Path "$root\runtimes\pnpm\bin\pnpm.cjs") }
if ($Runtimes -contains 'nub') { Check 'runtime nub runs' ((& "$root\runtimes\nub\nub.exe" --version 2>&1) -match '\d') }
if ($Runtimes -contains 'bun') { Check 'agent runtime path finds bun' ((& "$root\bin\agent.exe" runtime path bun 2>&1 | Select-Object -First 1) -eq "$root\runtimes\bun\bun.exe") }

'== rm and add again'
if ($Runtimes -contains 'bun') {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" rm bun 2>&1 | Select-Object -Last 1
    Check 'rm: bun is gone' (-not (Test-Path "$root\runtimes\bun"))
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" add bun 2>&1 | Select-Object -Last 1
    Check 'add: bun is back' (Test-Path "$root\runtimes\bun\bun.exe")
}

'== a download that does not match its sum is refused'
$good = Get-Content "$root\scripts\components.tsv"
$bad = @(); $done = $false
foreach ($l in $good) {
    $f = $l -split "`t"
    if (-not $done -and $f.Count -ge 10 -and $f[1] -eq 'uv' -and $f[2] -eq 'windows' -and $f[3] -eq 'x86_64') { $f[6] = ('0' * 64); $l = $f -join "`t"; $done = $true }
    $bad += $l
}
Set-Content -LiteralPath "$root\scripts\components.tsv" -Value $bad -Encoding ASCII
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" rm uv 2>&1 | Out-Null
$out = (& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" add uv 2>&1 | Out-String)
Check 'a wrong SHA-256 is refused' ($out -match 'mismatch') $out
Check 'a wrong SHA-256 installs nothing' (-not (Test-Path "$root\runtimes\uv"))
Set-Content -LiteralPath "$root\scripts\components.tsv" -Value $good -Encoding ASCII

'== uninstall --keep-data, then uninstall'
New-Item -ItemType Directory -Path "$root\models\llama\keep" -Force | Out-Null
Set-Content "$root\models\llama\keep\m.gguf" 'x'
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" uninstall --keep-data --yes 2>&1 | Select-Object -Last 2
Check 'keep-data: the program is gone' (-not (Test-Path "$root\bin") -and -not (Test-Path "$root\engines"))
Check 'keep-data: the models stay' (Test-Path "$root\models\llama\keep\m.gguf")
& powershell -NoProfile -ExecutionPolicy Bypass -File $installer install --yes 2>&1 | Out-Null
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" uninstall --yes 2>&1 | Select-Object -Last 1
Check 'uninstall: the folder is gone' (-not (Test-Path $root))
Check 'the user environment is still untouched' (([Environment]::GetEnvironmentVariable('AGENT_HOME', 'User') -eq $userHomeBefore) -and ([Environment]::GetEnvironmentVariable('Path', 'User') -eq $userPathBefore))
"SUMMARY $($script:pass) passed, $($script:fail) failed"
