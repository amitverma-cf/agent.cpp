# 04 custom engine

`reverse_engine.cpp` is a complete engine plugin (it reverses text and "draws" a gradient). `host.cpp` loads it, runs it in process and
under process isolation, and checks streaming, files, progress, cancel and refusal of tasks it does not declare. See
[writing an engine](../../docs/guides/writing-an-engine.md).
