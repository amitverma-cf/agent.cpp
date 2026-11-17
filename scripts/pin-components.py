#!/usr/bin/env python3
"""Writes scripts/components.tsv: the exact version of every component an agent.cpp release works with, with the SHA-256 of each file.

A maintainer runs this when raising the pin of an upstream project or before tagging a release; the installer scripts only read the result.

  python3 scripts/pin-components.py --tag v1.2.0 > scripts/components.tsv

Columns (tab separated, "-" for none):
  kind  name  os  arch  backend  version  sha256  strip  dest  url
kind: core (the program), engine (a plugin the program loads), runtime (a tool agent.cpp runs for MCP servers and skills).
backend: cpu, vulkan, cuda12, cuda13, rocm, metal, or any (belongs to every backend of that engine).
sha256: the sum; "@sums" for a file of the agent.cpp release itself (its SHA256SUMS holds the sum).
strip: how many single-folder levels of the archive to drop, or "find:FILE" (use the folder that holds FILE).
dest: a folder inside the component's folder; "." is the folder itself.
url: "@RELEASE@" is the address of the agent.cpp release and "@TAG@" its tag.

Every upstream version below is the one tested with this release. GitHub publishes a SHA-256 for each asset, which is what is recorded.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request

# ---- the pins: raise one here, run the script, test, release
PINS = {
    "bun": "1.4.4",
    "uv": "0.13.9",
    "nub": "0.11.1",
    "node": "v24",  # the newest v24.x at the time of pinning
    "pnpm": "10",  # the newest 10.x
    "onnxruntime": "1.32.0",
}

OURS_ENGINES = {  # engines built by agent.cpp's own CI: name -> backends built
    "llama": ["cpu", "vulkan"],
    "audiocpp": ["cpu", "vulkan"],
    "sdcpp": ["cpu", "vulkan"],
    "trellis": ["cpu", "vulkan"],
    "onnx": ["any"],
}
PLATFORMS = [("linux", "x86_64"), ("linux", "aarch64"), ("windows", "x86_64"), ("windows", "aarch64")]


def ext(os_):
    return "zip" if os_ == "windows" else "tar.gz"


def api(path):
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except (OSError, subprocess.CalledProcessError):
        req = urllib.request.Request("https://api.github.com/" + path, headers={"User-Agent": "pin-components"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


def release_assets(repo, tag):
    data = api("repos/%s/releases/tags/%s" % (repo, tag))
    out = {}
    for a in data["assets"]:
        d = a.get("digest") or ""
        if d.startswith("sha256:"):
            out[a["name"]] = (d[7:], a["browser_download_url"])
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pin-components"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


rows = []


def add(kind, name, os_, arch, backend, version, sha, strip, dest, url):
    rows.append("\t".join([kind, name, os_, arch, backend, version, sha, strip, dest, url]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="the agent.cpp release these pins belong to, e.g. v1.2.0")
    tag = ap.parse_args().tag

    # ---- our own files: the program and the engine plugins
    for os_, arch in PLATFORMS:
        add("core", "agent", os_, arch, "any", tag, "@sums", "0", ".", "@RELEASE@/agent-cpp-core-@TAG@-%s-%s.%s" % (os_, arch, ext(os_)))
        for eng, backends in OURS_ENGINES.items():
            for b in backends:
                if b == "vulkan" and arch != "x86_64" and not (os_ == "linux" and arch == "aarch64"):
                    continue  # the Vulkan builds are x86_64 on both systems and aarch64 on Linux
                suffix = "-vulkan" if b == "vulkan" else ""
                add("engine", eng, os_, arch, b, tag, "@sums", "0", ".",
                    "@RELEASE@/agent-engine-%s-@TAG@-%s-%s%s.%s" % (eng, os_, arch, suffix, ext(os_)))

    # ---- onnxruntime: the official library, next to our adapter
    v = PINS["onnxruntime"]
    assets = release_assets("microsoft/onnxruntime", "v" + v)
    ort = {("linux", "x86_64"): "linux-x64", ("linux", "aarch64"): "linux-aarch64", ("windows", "x86_64"): "win-x64", ("windows", "aarch64"): "win-arm64"}
    libname = {"linux": "libonnxruntime.so.1", "windows": "onnxruntime.dll"}
    for (os_, arch), t in ort.items():
        name = "onnxruntime-%s-%s.%s" % (t, v, "zip" if os_ == "windows" else "tgz")
        sha, url = assets[name]
        add("engine", "onnx", os_, arch, "any", v, sha, "find:" + libname[os_], ".", url)

    # ---- runtimes
    v = PINS["bun"]
    assets = release_assets("oven-sh/bun", "bun-v" + v)
    for (os_, arch), t in {("linux", "x86_64"): "linux-x64", ("linux", "aarch64"): "linux-aarch64", ("windows", "x86_64"): "windows-x64",
                           ("macos", "x86_64"): "darwin-x64", ("macos", "aarch64"): "darwin-aarch64"}.items():
        sha, url = assets["bun-%s.zip" % t]
        add("runtime", "bun", os_, arch, "any", v, sha, "find:" + ("bun.exe" if os_ == "windows" else "bun"), ".", url)

    v = PINS["uv"]
    assets = release_assets("astral-sh/uv", v)
    for (os_, arch), t in {("linux", "x86_64"): "x86_64-unknown-linux-gnu", ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
                           ("windows", "x86_64"): "x86_64-pc-windows-msvc", ("windows", "aarch64"): "aarch64-pc-windows-msvc",
                           ("macos", "x86_64"): "x86_64-apple-darwin", ("macos", "aarch64"): "aarch64-apple-darwin"}.items():
        sha, url = assets["uv-%s.%s" % (t, "zip" if os_ == "windows" else "tar.gz")]
        add("runtime", "uv", os_, arch, "any", v, sha, "find:" + ("uv.exe" if os_ == "windows" else "uv"), ".", url)

    v = PINS["nub"]
    assets = release_assets("nubjs/nub", "v" + v)
    for (os_, arch), t in {("linux", "x86_64"): "linux-x64", ("linux", "aarch64"): "linux-arm64", ("windows", "x86_64"): "win32-x64",
                           ("windows", "aarch64"): "win32-arm64", ("macos", "x86_64"): "darwin-x64", ("macos", "aarch64"): "darwin-arm64"}.items():
        name = "nub-%s.%s" % (t, "zip" if os_ == "windows" else "tar.gz")
        if name not in assets:
            continue
        sha, url = assets[name]
        add("runtime", "nub", os_, arch, "any", v, sha, "find:" + ("nub.exe" if os_ == "windows" else "nub"), ".", url)

    major = PINS["node"]
    base = "https://nodejs.org/dist/latest-%s.x/" % major
    sums = fetch(base + "SHASUMS256.txt").decode()
    nodev = None
    for (os_, arch), t in {("linux", "x86_64"): "linux-x64", ("linux", "aarch64"): "linux-arm64", ("windows", "x86_64"): "win-x64",
                           ("windows", "aarch64"): "win-arm64", ("macos", "x86_64"): "darwin-x64", ("macos", "aarch64"): "darwin-arm64"}.items():
        suffix = "-%s.%s" % (t, "zip" if os_ == "windows" else "tar.gz")
        for line in sums.splitlines():
            h, _, f = line.strip().partition("  ")
            if f.startswith("node-v") and f.endswith(suffix):
                nodev = f[len("node-"):-len(suffix)]
                # the pinned folder: a version, not "latest", so the file stays the same tomorrow
                add("runtime", "node", os_, arch, "any", nodev, h, "find:" + ("node.exe" if os_ == "windows" else "node"), ".",
                    "https://nodejs.org/dist/%s/%s" % (nodev, f))
                break

    pv = PINS["pnpm"]
    meta = json.loads(fetch("https://registry.npmjs.org/pnpm/latest-%s" % pv))
    tarball = meta["dist"]["tarball"]
    sha = hashlib.sha256(fetch(tarball)).hexdigest()
    for os_, arch in [(o, a) for o in ("linux", "windows", "macos") for a in ("x86_64", "aarch64")]:
        add("runtime", "pnpm", os_, arch, "any", meta["version"], sha, "1", ".", tarball)

    out = ["# agent.cpp components, pinned for release %s. Written by scripts/pin-components.py: do not edit by hand." % tag,
           "# tag %s" % tag,
           "# kind\tname\tos\tarch\tbackend\tversion\tsha256\tstrip\tdest\turl"] + rows
    sys.stdout.buffer.write(("\n".join(out) + "\n").encode("utf-8"))  # LF on every system: the shell script reads this file


if __name__ == "__main__":
    main()
