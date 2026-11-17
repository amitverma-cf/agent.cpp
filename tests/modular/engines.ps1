# Real models through the engines of a modular install on Windows. Installs into a folder of its own from a release folder and the real
# upstream downloads, then runs each installed engine on a real model. Nothing outside the folder is touched.
#
#   powershell -File engines.ps1 -Release DIR -Scripts DIR -ChatModel C:\path\model.gguf [-Tts C:\path\english] [-Engines llama,audiocpp,onnx]
param([string]$Release, [string]$Scripts, [string]$ChatModel, [string]$Tts = '', [string[]]$Engines = @('llama', 'audiocpp', 'onnx'), [string]$Work = "$env:TEMP\agent-engtest")
$ErrorActionPreference = 'Continue'
$Engines = @($Engines | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
if (Test-Path $Work) { Remove-Item -Recurse -Force $Work }
New-Item -ItemType Directory -Path $Work | Out-Null
$root = Join-Path $Work 'root'
$env:AGENT_HOME = $root; $env:AGENT_RELEASE = $Release; $env:AGENT_SCRIPTS = $Scripts; $env:AGENT_VERSION = $(if ($env:AGENT_TEST_TAG) { $env:AGENT_TEST_TAG } else { 'local' })
$env:AGENT_NO_ENV = '1'; $env:AGENT_YES = '1'; $env:AGENT_NO_START = '1'; $env:NO_COLOR = '1'
$pass = 0; $fail = 0
function Check([string]$name, [bool]$ok, [string]$why = '') { if ($ok) { $script:pass++; "PASS $name" } else { $script:fail++; "FAIL $name  ::  $why" } }
function Show($text, [int]$n = 4) { ($text -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last $n | ForEach-Object { '    ' + $_.Substring(0, [Math]::Min(200, $_.Length)) }) }

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Scripts 'install.ps1') install --yes 2>&1 | Select-Object -Last 3
foreach ($e in $Engines) { & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\install.ps1" add $e 2>&1 | Select-Object -Last 3 }
$agent = Join-Path $root 'bin\agent.exe'
'== doctor'
Show ((& $agent doctor 2>&1 | Out-String)) 12
foreach ($e in $Engines) { Check "engine ${e}: installed" (Test-Path "$root\engines\$e\.pin") }

if ($Engines -contains 'llama') {
    '== llama (upstream release b12020 + our plugin)'
    & $agent model pull $ChatModel 2>&1 | Out-Null
    $name = [IO.Path]::GetFileNameWithoutExtension($ChatModel)
    $out = (& $agent chat -m $name -y -p 'Reply with only the single word: pineapple' 2>&1 | Out-String)
    Show $out 3
    Check 'llama: a real chat answer comes back' ($out -match '(?i)pineapple') $out
    Check 'llama: the plugin is small and the libraries are upstream' ((Get-Item "$root\engines\llama\agent-engine-llama.dll").Length -lt 3MB -and (Test-Path "$root\engines\llama\llama.dll"))
}
if ($Engines -contains 'audiocpp' -and $Tts) {
    '== audiocpp (audio.cpp release library + our plugin)'
    $dest = Join-Path $root 'models\audiocpp\local'
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Copy-Item -Recurse -Force (Join-Path $Tts 'embeddings') $dest
    & $agent model pull (Join-Path $Tts 'pocket-tts-english-q8_0.gguf') 2>&1 | Out-Null
    $out = (& $agent task tts -m pocket-tts-english-q8_0 --text 'Hello there' 2>&1 | Out-String)
    Show $out 4
    Check 'audiocpp: speech is written to a wav file' ($out -match '(?i)wav|audio/') $out
    $wav = Get-ChildItem -Recurse -Filter *.wav $root -ErrorAction SilentlyContinue | Select-Object -First 1
    Check 'audiocpp: the wav file has sound in it' ($wav -and $wav.Length -gt 20000) "$($wav.FullName) $($wav.Length)"
}
if ($Engines -contains 'onnx') {
    '== onnx (the official onnxruntime)'
    $bytes = [byte[]](0x08,0x08,0x12,0x09,0x61,0x67,0x65,0x6e,0x74,0x2e,0x63,0x70,0x70,0x3a,0x63,0x0a,0x15,0x0a,0x01,0x61,0x0a,0x01,0x62,0x12,0x03,0x73,0x75,0x6d,0x1a,0x03,0x61,0x64,0x64,0x22,0x03,0x41,0x64,0x64,0x12,0x09,0x61,0x64,0x64,0x5f,0x67,0x72,0x61,0x70,0x68,0x5a,0x13,0x0a,0x01,0x61,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x5a,0x13,0x0a,0x01,0x62,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x62,0x15,0x0a,0x03,0x73,0x75,0x6d,0x12,0x0e,0x0a,0x0c,0x08,0x01,0x12,0x08,0x0a,0x02,0x08,0x02,0x0a,0x02,0x08,0x02,0x42,0x04,0x0a,0x00,0x10,0x0d)
    [IO.File]::WriteAllBytes("$Work\add.onnx", $bytes)
    & $agent model pull "$Work\add.onnx" 2>&1 | Out-Null
    $out = (& $agent task generic -e onnx -m add --tensor a:f32:2,2=1,2,3,4 --tensor b:f32:2,2=10,20,30,40 2>&1 | Out-String)
    Show $out 4
    Check 'onnx: a real model adds two tensors (11 22 33 44)' ($out -match '11') $out
    Check 'onnx: the library is the official onnxruntime' (Test-Path "$root\engines\onnx\onnxruntime.dll")
}
"SUMMARY $pass passed, $fail failed"
