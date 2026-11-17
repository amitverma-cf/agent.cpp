# Removes agent.cpp from Windows. `agent uninstall` runs this file after it has asked; it is given what to remove through the environment:
#   AGENT_UNINSTALL_HOME      the folder of the data (models, engines, conversations, ...); empty = leave the data
#   AGENT_UNINSTALL_PROGRAM   the folder of the program (the one on PATH)
#   AGENT_UNINSTALL_POINTER   the file that names the home (home.json), when it is outside both folders
# It waits for the program that started it to exit (Windows will not delete a running program), removes the folders, takes the program
# folder out of the user PATH and removes the pointer file.
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest)
$ErrorActionPreference = 'Continue'
if ($env:AGENT_WAIT_PID) { Wait-Process -Id ([int]$env:AGENT_WAIT_PID) -Timeout 30 -ErrorAction SilentlyContinue }

function Step($text) { Write-Host '  ' -NoNewline; Write-Host ([string][char]0x2713) -ForegroundColor Cyan -NoNewline; Write-Host " $text" }
function Info($text) { Write-Host "  $([char]0x00B7) $text" -ForegroundColor DarkGray }

function Remove-Folder($path) {
    if (-not $path -or -not (Test-Path -LiteralPath $path)) { return }
    # A path this short is a drive or the user folder: never removed.
    if ($path.TrimEnd('\').Length -lt 8 -or $path.TrimEnd('\') -eq $env:USERPROFILE) { Info "Not removing $path"; return }
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $path) { Info "Some files in $path are in use; delete the folder after closing programs that use it" }
    else { Step "Removed $path" }
}

Write-Host ''
Write-Host '  agent.cpp  uninstall' -ForegroundColor Cyan
Write-Host ''
$program = $env:AGENT_UNINSTALL_PROGRAM
$data = $env:AGENT_UNINSTALL_HOME
Remove-Folder $data
if ($program -and $data -and $program.StartsWith($data, [StringComparison]::OrdinalIgnoreCase)) { } else { Remove-Folder $program }
if ($env:AGENT_UNINSTALL_POINTER -and (Test-Path -LiteralPath $env:AGENT_UNINSTALL_POINTER)) {
    Remove-Item -LiteralPath $env:AGENT_UNINSTALL_POINTER -Force -ErrorAction SilentlyContinue
    Step "Removed $($env:AGENT_UNINSTALL_POINTER)"
}
if ($program) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $kept = (($userPath -split ';') | Where-Object { $_ -and $_.TrimEnd('\') -ne $program.TrimEnd('\') }) -join ';'
    if ($kept -ne $userPath) {
        [Environment]::SetEnvironmentVariable('Path', $kept, 'User')
        Step 'Removed agent from your PATH'
    }
}
Write-Host ''
Write-Host '  agent.cpp is removed.' -ForegroundColor Cyan
Write-Host ''
Start-Sleep -Seconds 2
