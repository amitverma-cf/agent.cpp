# agent.cpp on Windows: installs, sets up, updates, extends and removes it. This one script does all of it; the program only hands over to it
# (agent setup, agent update, agent uninstall, agent engine pull ...).
#
#   irm https://agent.qozu.fyi/install.ps1 | iex                       install (asks where, once), then set up
#   & ([scriptblock]::Create((irm https://agent.qozu.fyi/install.ps1))) add llama
#
#   install.ps1 [install]            a new install: the folder, the program, then setup
#   install.ps1 setup                engines, runtimes and a model; can be run again at any time
#   install.ps1 add NAME[,NAME]      add engines or runtimes (llama, sdcpp, audiocpp, onnx, trellis, bun, uv, node, pnpm, nub, ...)
#   install.ps1 rm NAME[,NAME]       remove them
#   install.ps1 ls                   what exists for this machine and what is installed
#   install.ps1 update [--check]     move to the newest agent.cpp release and the components it pins
#   install.ps1 uninstall            remove everything (--keep-data keeps models, notes and settings)
#
# One folder holds everything: bin\ (the program), engines\, runtimes\, models\, workspaces\, scripts\ (this file and components.tsv, of the
# installed release), cache\. Its path is the user environment variable AGENT_HOME, which is also what the program reads; <folder>\bin is
# on the user PATH. Nothing else is written outside the folder.
#
# Pinned: a release of agent.cpp lists the exact version of every component it works with (components.tsv). add, setup and rm use the list
# of the installed release; only update reads a newer one. Without arguments (irm | iex) the options come from environment variables:
# AGENT_CMD, AGENT_HOME, AGENT_VERSION (a tag), AGENT_BACKEND (cpu, vulkan, cuda12, cuda13, rocm), AGENT_YES, AGENT_ENGINES, AGENT_RUNTIMES,
# AGENT_REPO, AGENT_SCRIPTS and AGENT_RELEASE (an address or folder to read instead of GitHub), AGENT_NO_ENV (leave the user environment alone).
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Step($text) { Write-Host '  ' -NoNewline; Write-Host ([string][char]0x2713) -ForegroundColor Cyan -NoNewline; Write-Host " $text" }
function Info($text) { Write-Host "  $([char]0x00B7) $text" -ForegroundColor DarkGray }
function Fail($text) { throw $text }

function Invoke-AgentInstaller([string[]] $Argv) {
    $script:Repo = if ($env:AGENT_REPO) { $env:AGENT_REPO } else { 'amitverma-cf/agent.cpp' }
    $cmd = if ($env:AGENT_CMD) { $env:AGENT_CMD } else { '' }
    $yes = [bool]$env:AGENT_YES; $backend = $env:AGENT_BACKEND; $engines = $env:AGENT_ENGINES; $runtimes = $env:AGENT_RUNTIMES
    $model = ''; $noModel = $false; $noEngines = $false; $noRuntimes = $false; $check = $false; $keepData = $false; $all = $false
    $version = $env:AGENT_VERSION; $names = @()
    $i = 0
    while ($i -lt $Argv.Count) {
        $a = $Argv[$i]
        switch -Regex ($a) {
            '^(--yes|-y)$' { $yes = $true; break }
            '^--backend$' { $i++; $backend = $Argv[$i]; break }
            '^--engines$' { $i++; $engines = $Argv[$i]; break }
            '^--runtimes$' { $i++; $runtimes = $Argv[$i]; break }
            '^--all$' { $all = $true; break }
            '^--no-engines$' { $noEngines = $true; break }
            '^--no-runtimes$' { $noRuntimes = $true; break }
            '^--model$' { $i++; $model = $Argv[$i]; break }
            '^--no-models?$' { $noModel = $true; break }
            '^--check$' { $check = $true; break }
            '^--keep-data$' { $keepData = $true; break }
            '^--(version|tag)$' { $i++; $version = $Argv[$i]; break }
            '^--home$' { $i++; $env:AGENT_HOME = $Argv[$i]; break }
            '^-' { Fail "unknown option $a" }
            default { if (-not $cmd) { $cmd = $a } else { $names += $a }; break }
        }
        $i++
    }
    if (-not $cmd) { $cmd = 'install' }
    $nameList = ($names -join ',').Trim(',')

    # `agent update` and `agent uninstall` start this script and exit: Windows will not replace a running program, so wait for it.
    if ($env:AGENT_WAIT_PID) { Wait-Process -Id ([int]$env:AGENT_WAIT_PID) -Timeout 30 -ErrorAction SilentlyContinue }

    $ask = (-not $yes) -and [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    function Prompt([string]$question, [string]$default) {
        $answer = Read-Host "  $question"
        if ($answer) { $answer.Trim() } else { $default }
    }
    function Confirm-Yes([string]$question) {
        if ($yes) { return $true }
        if (-not $ask) { return $false }
        return (Prompt "$question [y/N]" 'n') -match '^(y|yes)$'
    }

    # ---- this machine
    $os = 'windows'
    $arch = switch ($env:PROCESSOR_ARCHITEW6432 + $env:PROCESSOR_ARCHITECTURE) {
        { $_ -match 'ARM64' } { 'aarch64' }
        { $_ -match 'AMD64' } { 'x86_64' }
        default { Fail "Unsupported CPU: $($env:PROCESSOR_ARCHITECTURE). x64 and arm64 are supported." }
    }
    function Get-Backends {
        $list = @()
        $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($smi -and $arch -eq 'x86_64') {
            $text = (& $smi.Source 2>$null | Out-String)
            if ($text -match 'CUDA Version:\s*(\d+)') { if ([int]$Matches[1] -ge 13) { $list += 'cuda13', 'cuda12' } else { $list += 'cuda12' } }
        }
        if (Test-Path "$env:SystemRoot\System32\vulkan-1.dll") { $list += 'vulkan' }
        $list += 'cpu'
        return $list
    }
    $backends = if ($backend) { @($backend, 'cpu') } else { Get-Backends }

    $script:tmp = $null; $script:ROOT = $null; $script:BIN = $null
    function Get-Source([string]$src, [string]$dst, [bool]$progress = $false) {
        if ($src -match '^https?://') {
            if ($progress) {
                Add-Type -AssemblyName System.Net.Http
                $client = New-Object System.Net.Http.HttpClient
                $client.Timeout = [TimeSpan]::FromMinutes(60)
                $r = $client.GetAsync($src, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
                [void]$r.EnsureSuccessStatusCode()
                $total = $r.Content.Headers.ContentLength
                $in = $r.Content.ReadAsStreamAsync().Result
                $out = [System.IO.File]::Create($dst)
                $buf = New-Object byte[] 262144
                $done = 0; $shown = -1
                while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                    $out.Write($buf, 0, $n); $done += $n
                    if ($total) {
                        $pct = [int](100 * $done / $total)
                        if ($pct -ne $shown) {
                            $shown = $pct; $filled = [int]($pct / 4)
                            Write-Host ("`r    [" + ('#' * $filled) + (' ' * (25 - $filled)) + "] $pct%  " + [math]::Round($done / 1MB, 1) + ' / ' + [math]::Round($total / 1MB, 1) + ' MB') -NoNewline -ForegroundColor DarkGray
                        }
                    }
                }
                $out.Dispose(); $in.Dispose(); $client.Dispose()
                Write-Host ''
            } else {
                Invoke-WebRequest -UseBasicParsing -Uri $src -OutFile $dst
            }
        } else {
            Copy-Item -LiteralPath ($src -replace '^file:///?', '') -Destination $dst -Force
        }
    }

    # ---- the folder
    function Resolve-Root {
        $r = $env:AGENT_HOME
        if (-not $r -and $cmd -eq 'install') {
            $def = Join-Path $env:USERPROFILE '.agent.cpp'
            if ($ask) {
                Write-Host ''
                Write-Host '  Where should agent.cpp keep everything?  The program, engines, models, memory and settings all go in this one folder.'
                $r = Prompt "Folder [$def]:" $def
            } else { $r = $def }
        }
        if (-not $r) { Fail 'AGENT_HOME is not set: agent.cpp is not installed here. Install it first: irm https://agent.qozu.fyi/install.ps1 | iex' }
        $r = $r.Trim().Trim('"').Trim("'")
        if ($r -eq '~') { $r = $env:USERPROFILE } elseif ($r.StartsWith('~\') -or $r.StartsWith('~/')) { $r = Join-Path $env:USERPROFILE $r.Substring(2) }
        if (-not [System.IO.Path]::IsPathRooted($r) -or $r -notmatch '^([A-Za-z]:[\\/]|\\\\)') { Fail "The folder must be a full path such as C:\agent, not '$r'." }
        $r = [System.IO.Path]::GetFullPath($r).TrimEnd('\')
        if ($r.Length -ge 2 -and $r[1] -eq ':') { $r = $r.Substring(0, 1).ToUpper() + $r.Substring(1) } # c:\x and C:\x are one folder
        if ($r -eq $env:USERPROFILE.TrimEnd('\') -or $r.Length -le 3) { Fail 'Choose a folder of its own, not your user folder or a drive.' }
        if ((Test-Path -LiteralPath $r -PathType Leaf)) { Fail "$r is a file." }
        if ($cmd -eq 'install' -and (Test-Path -LiteralPath $r) -and -not (Test-Path -LiteralPath (Join-Path $r '.agent-root')) -and (Get-ChildItem -LiteralPath $r -Force | Select-Object -First 1)) {
            Fail "$r already has other files in it. Choose an empty folder or a new one."
        }
        New-Item -ItemType Directory -Path $r -Force | Out-Null
        $script:ROOT = $r
        $script:BIN = Join-Path $r 'bin'
        $env:AGENT_HOME = $r
    }

    # The folder is remembered as the user environment variable AGENT_HOME, and <folder>\bin as a user PATH entry.
    function Remove-Env([string]$oldBin) {
        if ($env:AGENT_NO_ENV) { return }
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if ($userPath) {
            $kept = (($userPath -split ';') | Where-Object { $_ -and $_.TrimEnd('\') -ne $oldBin.TrimEnd('\') }) -join ';'
            if ($kept -ne $userPath) { [Environment]::SetEnvironmentVariable('Path', $kept, 'User') }
        }
    }
    function Set-Env {
        if ($env:AGENT_NO_ENV) { Info 'AGENT_NO_ENV is set: the user environment is left alone'; return }
        $old = [Environment]::GetEnvironmentVariable('AGENT_HOME', 'User')
        if ($old -and $old.TrimEnd('\') -ne $ROOT) { Remove-Env (Join-Path $old 'bin') }
        [Environment]::SetEnvironmentVariable('AGENT_HOME', $ROOT, 'User')
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not $userPath) { $userPath = '' }
        if (($userPath -split ';') -notcontains $BIN) { [Environment]::SetEnvironmentVariable('Path', (($userPath.TrimEnd(';') + ';' + $BIN).TrimStart(';')), 'User') }
        Step "AGENT_HOME is $ROOT, and $BIN is on PATH (open a new terminal)"
    }

    function New-Tmp {
        $d = Join-Path $ROOT 'cache\tmp'
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        $script:tmp = Join-Path $d ('install.' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        New-Item -ItemType Directory -Path $script:tmp -Force | Out-Null
    }

    # ---- the release and the list of components
    function Get-ReleaseBase([string]$tag) {
        if ($env:AGENT_RELEASE) { return $env:AGENT_RELEASE }
        if ($tag) { return "https://github.com/$Repo/releases/download/$tag" }
        return "https://github.com/$Repo/releases/latest/download"
    }
    function Get-LatestTag {
        $r = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$Repo/releases/latest"
        return $r.tag_name
    }
    # Columns (tab separated): kind name os arch backend version sha256 strip dest url. sha256 "@sums": the sum is in the release's SHA256SUMS.
    $script:List = $null
    function Read-List([string]$file) {
        $script:ListFile = $file
        $script:List = @(Get-Content -LiteralPath $file | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object {
            $f = $_ -split "`t"
            [pscustomobject]@{ Kind = $f[0]; Name = $f[1]; Os = $f[2]; Arch = $f[3]; Backend = $f[4]; Version = $f[5]; Sha = $f[6]; Strip = $f[7]; Dest = $f[8]; Url = $f[9] }
        })
    }
    function Get-ListTag { $m = Select-String -LiteralPath $script:ListFile -Pattern '^# tag (.+)$' | Select-Object -First 1; if ($m) { $m.Matches[0].Groups[1].Value.Trim() } else { '' } }
    function Load-List {
        $f = Join-Path $ROOT 'scripts\components.tsv'
        if (Test-Path -LiteralPath $f) { Read-List $f; return }
        if (-not $tmp) { New-Tmp }
        $base = if ($env:AGENT_SCRIPTS) { $env:AGENT_SCRIPTS } else { "https://raw.githubusercontent.com/$Repo/$(if ($version) { $version } else { 'main' })/scripts" }
        $dst = Join-Path $tmp 'components.tsv'
        Get-Source "$base/components.tsv" $dst
        Read-List $dst
    }
    function Get-Rows([string]$kind, [string]$name) { $script:List | Where-Object { $_.Kind -eq $kind -and $_.Name -eq $name -and $_.Os -eq $os -and $_.Arch -eq $arch } }
    function Get-KindOf([string]$name) { $r = $script:List | Where-Object { $_.Name -eq $name } | Select-Object -First 1; if ($r) { $r.Kind } else { '' } }
    function Pick-Backend([string]$kind, [string]$name) {
        $rows = @(Get-Rows $kind $name)
        if (-not $rows) { return '' }
        $have = @($rows | ForEach-Object { $_.Backend } | Sort-Object -Unique)
        if ($have.Count -eq 1 -and $have[0] -eq 'any') { return 'any' }
        foreach ($b in $backends) { if ($have -contains $b) { return $b } }
        return ''
    }
    function Get-Dir([string]$kind, [string]$name) {
        switch ($kind) { 'engine' { Join-Path $ROOT "engines\$name" } 'runtime' { Join-Path $ROOT "runtimes\$name" } default { $BIN } }
    }
    $script:Sums = $null
    function Get-Sum([string]$sha, [string]$file, [string]$rel) {
        if ($sha -ne '@sums') { return $sha.ToLower() }
        if (-not $script:Sums) {
            $f = Join-Path $tmp 'SHA256SUMS'
            Get-Source "$rel/SHA256SUMS" $f
            $script:Sums = @(Get-Content -LiteralPath $f)
        }
        $line = $script:Sums | Where-Object { $_ -match "\s\*?$([regex]::Escape($file))\s*$" } | Select-Object -First 1
        if (-not $line) { Fail "$file is not in SHA256SUMS." }
        return ($line.Trim() -split '\s+')[0].ToLower()
    }
    function Expand-Component([string]$archive, [string]$dest, [string]$strip) {
        $x = Join-Path $tmp ('x' + [guid]::NewGuid().ToString('N').Substring(0, 6))
        New-Item -ItemType Directory -Path $x -Force | Out-Null
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        if ($archive -match '\.zip$') { Expand-Archive -LiteralPath $archive -DestinationPath $x -Force }
        elseif ($archive -match '\.(tar\.gz|tgz)$') { & (Join-Path $env:SystemRoot 'System32\tar.exe') -xzf $archive -C $x; if ($LASTEXITCODE -ne 0) { Fail "Cannot unpack $(Split-Path $archive -Leaf)." } }
        else { Fail "Unknown archive type: $archive" }
        $src = $x
        if ($strip -like 'find:*') {
            $name = $strip.Substring(5)
            $hit = Get-ChildItem -LiteralPath $x -Recurse -File -Depth 4 -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
            if (-not $hit) { Fail "$(Split-Path $archive -Leaf) does not contain $name." }
            $src = $hit.DirectoryName
        } else {
            for ($k = 0; $k -lt [int]$strip; $k++) {
                $children = @(Get-ChildItem -LiteralPath $src -Force)
                if ($children.Count -ne 1 -or -not $children[0].PSIsContainer) { break }
                $src = $children[0].FullName
            }
        }
        Copy-Item -Path (Join-Path $src '*') -Destination $dest -Recurse -Force
        Remove-Item -LiteralPath $x -Recurse -Force -ErrorAction SilentlyContinue
    }
    function Install-Component([string]$kind, [string]$name, [string]$bk, [string]$rel, [string]$tag) {
        $final = Get-Dir $kind $name
        $stage = if ($kind -eq 'core') { Join-Path $ROOT 'bin.new' } else { "$final.new" }
        if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        $rows = @(Get-Rows $kind $name | Where-Object { $_.Backend -eq 'any' -or $_.Backend -eq $bk })
        if (-not $rows) { Fail "$name has no build for $os $arch." }
        $got = @()
        foreach ($r in $rows) {
            $url = $r.Url.Replace('@RELEASE@', $rel).Replace('@TAG@', $tag)
            $file = $url.Substring($url.LastIndexOfAny(@('/', '\')) + 1)
            Write-Host "  $([char]0x2193) $name  ($($r.Backend), $($r.Version))" -ForegroundColor Cyan
            $local = Join-Path $tmp $file
            try { Get-Source $url $local $true } catch { Fail "The download of $file failed: $url" }
            $want = Get-Sum $r.Sha $file $rel
            if ((Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLower() -ne $want) { Fail "SHA-256 mismatch for $file. Nothing was installed." }
            $dd = if ($r.Dest -eq '.' -or $r.Dest -eq '-') { $stage } else { Join-Path $stage $r.Dest }
            Expand-Component $local $dd $r.Strip
            Remove-Item -LiteralPath $local -Force
            $got += $r.Version
        }
        if ($kind -eq 'core') {
            $old = Join-Path $ROOT 'bin.old'
            if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
            if (Test-Path -LiteralPath $BIN) { Move-Item -LiteralPath $BIN -Destination $old }
            Move-Item -LiteralPath $stage -Destination $BIN
            if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction SilentlyContinue }
        } else {
            Set-Content -LiteralPath (Join-Path $stage '.pin') -Value @($bk, ($got -join ' ')) -Encoding ASCII
            if (Test-Path -LiteralPath $final) { Remove-Item -LiteralPath $final -Recurse -Force }
            New-Item -ItemType Directory -Path (Split-Path $final) -Force | Out-Null
            Move-Item -LiteralPath $stage -Destination $final
        }
        Step "$name installed ($bk)"
    }
    function Get-Pinned([string]$kind, [string]$name) {
        $p = Join-Path (Get-Dir $kind $name) '.pin'
        if (Test-Path -LiteralPath $p) { $l = @(Get-Content -LiteralPath $p); return "$($l[0]) $($l[1])" }
        return ''
    }
    function Get-Names([string]$kind) { @($script:List | Where-Object { $_.Kind -eq $kind -and $_.Os -eq $os -and $_.Arch -eq $arch } | ForEach-Object { $_.Name } | Sort-Object -Unique) }

    function Add-Names([string]$list, [string]$rel, [string]$tag) {
        if (-not $list) { return }
        if (-not $tmp) { New-Tmp }
        foreach ($n in ($list -split '[,\s]+' | Where-Object { $_ })) {
            $k = Get-KindOf $n
            if (-not $k -or $k -eq 'core') { Info "$n is not a component of this release (agent ls shows them)"; continue }
            $b = Pick-Backend $k $n
            if (-not $b) { Info "$n has no build for $os $arch yet"; continue }
            Install-Component $k $n $b $rel $tag
            if ($k -eq 'engine' -and (Test-Path -LiteralPath (Join-Path $BIN 'agent.exe'))) {
                $doc = (& (Join-Path $BIN 'agent.exe') doctor 2>$null | Out-String)
                if ($doc -match "(?m)^\s+$([regex]::Escape($n))\s+ok") { Step "$n loads" } else { Info "$n is installed but did not report ok in agent doctor" }
            }
        }
    }

    # ---------------------------------------------------------------------------------------------------------------- commands
    function Do-Setup {
        Resolve-Root; Load-List; if (-not $tmp) { New-Tmp }
        $tag = Get-ListTag; $rel = Get-ReleaseBase $tag
        Write-Host ''
        Write-Host '  agent.cpp' -ForegroundColor Cyan -NoNewline; Write-Host '  setup' -ForegroundColor DarkGray
        Write-Host "  $([string]([char]0x2500) * 28)" -ForegroundColor DarkGray
        Write-Host ''
        Step "Files are kept in $ROOT"
        $avail = Get-Names 'engine'
        $chosen = $engines
        if ($all) { $chosen = $avail -join ',' }
        if (-not $chosen -and -not $noEngines) {
            if ($ask) {
                Write-Host "  Engines available for this machine: $($avail -join ', ')"
                $chosen = Prompt 'Install which? (names separated by commas, Enter for llama, all for every one):' 'llama'
                if ($chosen -eq 'all') { $chosen = $avail -join ',' }
            } else { $chosen = 'llama' }
        }
        if ($noEngines) { $chosen = '' }
        Add-Names $chosen $rel $tag

        $avail = Get-Names 'runtime'
        $chosen = $runtimes
        if (-not $chosen -and -not $noRuntimes) {
            if ($ask) {
                Write-Host "  Runtimes (for MCP servers and skills): $($avail -join ', ')"
                $chosen = Prompt 'Install which? (Enter for bun,uv; none to skip):' 'bun,uv'
                if ($chosen -eq 'none') { $chosen = '' }
            } else { $chosen = 'bun,uv' }
        }
        if ($noRuntimes) { $chosen = '' }
        Add-Names $chosen $rel $tag

        $exe = Join-Path $BIN 'agent.exe'
        if (-not $model -and -not $noModel -and $ask) { $model = Prompt 'Model to download now: a Hugging Face link, owner/repo, or a file on this computer (Enter to skip):' '' }
        $model = $model.Trim().Trim('"').Trim("'")
        if ($model) {
            Write-Host "  $([char]0x2193) Adding $model" -ForegroundColor Cyan
            & $exe model pull $model
            if ($LASTEXITCODE -ne 0) { Fail 'The model could not be added.' }
        }
        $listed = if (Test-Path -LiteralPath $exe) { (& $exe model ls 2>$null | Out-String).Trim() } else { '' }
        $haveModel = $listed -and $listed -notmatch '^no models'
        Write-Host ''
        if ($haveModel -and $ask) { Write-Host '  Done. Starting the chat ...'; Write-Host ''; & $exe }
        elseif ($haveModel) { Write-Host '  Done. Run ' -NoNewline; Write-Host 'agent' -ForegroundColor Cyan -NoNewline; Write-Host ' to start the chat.' }
        else { Write-Host '  Done. Add a model with ' -NoNewline; Write-Host 'agent model pull <Hugging Face link or file>' -ForegroundColor Cyan -NoNewline; Write-Host ', then run ' -NoNewline; Write-Host 'agent' -ForegroundColor Cyan }
        Write-Host ''
    }

    function Do-Install {
        Resolve-Root
        Write-Host ''
        Write-Host '  agent.cpp' -ForegroundColor Cyan -NoNewline; Write-Host '  installer' -ForegroundColor DarkGray
        Write-Host "  $([string]([char]0x2500) * 28)" -ForegroundColor DarkGray
        Write-Host ''
        New-Tmp
        $tag = $version
        if (-not $tag -and -not $env:AGENT_RELEASE) { $tag = Get-LatestTag }
        if (-not $tag) { $tag = 'local' }
        $rel = Get-ReleaseBase $tag
        New-Item -ItemType Directory -Path (Join-Path $ROOT 'scripts') -Force | Out-Null
        $sbase = if ($env:AGENT_SCRIPTS) { $env:AGENT_SCRIPTS } else { "https://raw.githubusercontent.com/$Repo/$tag/scripts" }
        # The scripts and the list of components of this release are kept in the folder: every later command uses these, so what is
        # installed stays pinned until update.
        Get-Source "$sbase/components.tsv" (Join-Path $ROOT 'scripts\components.tsv')
        Get-Source "$sbase/install.ps1" (Join-Path $ROOT 'scripts\install.ps1')
        Read-List (Join-Path $ROOT 'scripts\components.tsv')
        Step "$os $arch  $([char]0x00B7)  backends: $($backends -join ' ')"
        Step "Release $(Get-ListTag)"
        Install-Component 'core' 'agent' 'any' $rel $tag
        New-Item -ItemType File -Path (Join-Path $ROOT '.agent-root') -Force | Out-Null
        foreach ($d in 'engines', 'runtimes', 'models', 'workspaces', 'cache', 'logs', 'skills', 'mcp', 'plugins') { New-Item -ItemType Directory -Path (Join-Path $ROOT $d) -Force | Out-Null }
        Set-Env
        $env:Path = "$BIN;$env:Path"
        if (-not $env:AGENT_NO_START) { Do-Setup } else { Write-Host ''; Write-Host '  Next: ' -NoNewline; Write-Host 'agent setup' -ForegroundColor Cyan; Write-Host '' }
    }

    # fetch NAME DIR: only the upstream parts of an engine (what its plugin is compiled against), for the release workflow. Everything is
    # unpacked whole, the header folders included, and no agent.cpp folder is needed.
    function Do-Fetch {
        $n = $names[0]; $d = $names[1]
        if (-not $n -or -not $d) { Fail 'usage: install.ps1 fetch NAME DIR' }
        $d = [System.IO.Path]::GetFullPath($d)
        $script:ROOT = Join-Path ([System.IO.Path]::GetTempPath()) ('agent-fetch-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
        New-Item -ItemType Directory -Path $script:ROOT -Force | Out-Null
        New-Tmp
        $list = if ($env:AGENT_LIST) { $env:AGENT_LIST } else { Join-Path $env:AGENT_SCRIPTS 'components.tsv' }
        Read-List $list
        $b = Pick-Backend 'engine' $n
        if (-not $b) { Fail "$n has no build for $os $arch." }
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        $fetched = $false
        foreach ($r in (Get-Rows 'engine' $n | Where-Object { ($_.Backend -eq 'any' -or $_.Backend -eq $b) -and -not $_.Url.StartsWith('@RELEASE@') })) {
            $file = $r.Url.Substring($r.Url.LastIndexOf('/') + 1)
            $local = Join-Path $tmp $file
            Get-Source $r.Url $local $true
            if ((Get-FileHash -Algorithm SHA256 -LiteralPath $local).Hash.ToLower() -ne $r.Sha.ToLower()) { Fail "SHA-256 mismatch for $file." }
            $st = if ($r.Strip -like 'find:*') { '0' } else { $r.Strip }  # whole archive: the header folders are wanted too
            Expand-Component $local $d $st
            Remove-Item -LiteralPath $local -Force
            Step "fetched $file ($($r.Version))"
            $fetched = $true
        }
        Remove-Item -LiteralPath $script:ROOT -Recurse -Force -ErrorAction SilentlyContinue
        $script:tmp = $null
        if (-not $fetched) { Fail "$n has no upstream part to fetch in scripts/components.tsv." }
    }

    function Do-Add {
        Resolve-Root; Load-List; if (-not $tmp) { New-Tmp }
        $tag = Get-ListTag
        if (-not $nameList) { Fail "Name what to add: $((Get-Names 'engine') -join ', '), $((Get-Names 'runtime') -join ', ')" }
        Add-Names $nameList (Get-ReleaseBase $tag) $tag
    }

    function Do-Rm {
        Resolve-Root; Load-List
        if (-not $nameList) { Fail 'Name what to remove.' }
        foreach ($n in ($nameList -split '[,\s]+' | Where-Object { $_ })) {
            $k = Get-KindOf $n
            $d = if ($k -and $k -ne 'core') { Get-Dir $k $n } else { '' }
            if ($d -and (Test-Path -LiteralPath $d)) { Remove-Item -LiteralPath $d -Recurse -Force; Step "$n removed" } else { Info "$n is not installed" }
        }
    }

    function Do-Ls {
        Resolve-Root; Load-List
        Write-Host ''
        Write-Host "  agent.cpp $(Get-ListTag)  $([char]0x00B7)  $os $arch  $([char]0x00B7)  backends: $($backends -join ' ')"
        Write-Host ''
        foreach ($k in 'engine', 'runtime') {
            foreach ($n in (Get-Names $k)) {
                $p = Get-Pinned $k $n
                if ($p) { Write-Host ('  {0,-10} {1,-9} installed  {2}' -f $n, $k, $p) }
                else { Write-Host ('  {0,-10} {1,-9} available  ({2})' -f $n, $k, (Pick-Backend $k $n)) }
            }
        }
        Write-Host ''
    }

    function Do-Update {
        Resolve-Root
        $f = Join-Path $ROOT 'scripts\components.tsv'
        if (-not (Test-Path -LiteralPath $f)) { Fail "agent.cpp is not installed in $ROOT." }
        Read-List $f; $have = Get-ListTag
        New-Tmp
        $tag = $version
        if (-not $tag) { $tag = Get-LatestTag }
        if ($tag -eq $have) { Step "Up to date ($have)"; return }
        if ($check) { Step "An update is available: $have -> $tag"; return }
        $rel = Get-ReleaseBase $tag
        $sbase = if ($env:AGENT_SCRIPTS) { $env:AGENT_SCRIPTS } else { "https://raw.githubusercontent.com/$Repo/$tag/scripts" }
        Get-Source "$sbase/components.tsv" (Join-Path $tmp 'new.tsv')
        Get-Source "$sbase/install.ps1" (Join-Path $tmp 'new.ps1')
        Step "Updating $have -> $tag"
        Copy-Item -LiteralPath (Join-Path $tmp 'new.tsv') -Destination $f -Force
        Read-List $f
        Install-Component 'core' 'agent' 'any' $rel $tag
        foreach ($k in 'engine', 'runtime') {
            foreach ($n in (Get-Names $k)) {
                if (-not (Get-Pinned $k $n)) { continue }
                $b = Pick-Backend $k $n
                if (-not $b) { Info "$n has no build in ${tag}: kept as it is"; continue }
                Install-Component $k $n $b $rel $tag
            }
        }
        Copy-Item -LiteralPath (Join-Path $tmp 'new.ps1') -Destination (Join-Path $ROOT 'scripts\install.ps1') -Force
        Step "Now at $tag"
    }

    function Do-Uninstall {
        Resolve-Root
        Write-Host ''
        Write-Host '  agent uninstall removes:'
        if ($keepData) { Write-Host "    the program, engines, runtimes and caches in $ROOT"; Write-Host '    (models, workspaces, settings and notes stay)' }
        else { Write-Host "    the whole folder $ROOT  (program, engines, runtimes, models, conversations, notes, logs)" }
        Write-Host '    AGENT_HOME and the PATH entry in your user environment'
        Write-Host ''
        if (-not (Confirm-Yes 'This cannot be undone. Remove it?')) { Info 'Nothing was removed (--yes removes without asking)'; return }
        Set-Location $env:USERPROFILE
        if ($keepData) {
            foreach ($d in 'bin', 'engines', 'runtimes', 'cache', 'scripts') { Remove-Item -LiteralPath (Join-Path $ROOT $d) -Recurse -Force -ErrorAction SilentlyContinue }
        } else {
            $script:tmp = $null
            Remove-Item -LiteralPath $ROOT -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $ROOT) { Info "Some files in $ROOT are in use; delete the folder after closing programs that use it" }
        }
        Remove-Env $BIN
        if (-not $env:AGENT_NO_ENV) { [Environment]::SetEnvironmentVariable('AGENT_HOME', $null, 'User') }
        Step 'agent.cpp is removed'
    }

    try {
        switch ($cmd) {
            'fetch' { Do-Fetch }
            'install' { Do-Install }
            'setup' { Do-Setup }
            'add' { Do-Add }
            { $_ -in 'rm', 'remove' } { Do-Rm }
            { $_ -in 'ls', 'list' } { Do-Ls }
            { $_ -in 'update', 'upgrade' } { Do-Update }
            'uninstall' { Do-Uninstall }
            default { Fail "unknown command '$cmd' (install, setup, add, rm, ls, update, uninstall)" }
        }
    } finally {
        if ($tmp -and (Test-Path -LiteralPath $tmp)) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
    }
}

try {
    Invoke-AgentInstaller $args
} catch {
    Write-Host ''
    Write-Host "  $([char]0x2717) $($_.Exception.Message)" -ForegroundColor Magenta
    Write-Host ''
    if ($PSCommandPath) { exit 1 }
}
