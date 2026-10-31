/* miniaudio implementation for the agent CLI's sound devices (audio.cpp).
 *
 * vendor/miniaudio/miniaudio.h is the single-header library from https://github.com/mackron/miniaudio,
 * release 0.11.26 (public domain or MIT-0, see vendor/miniaudio/LICENSE). Only the device I/O part is
 * compiled in; backends (ALSA, PulseAudio, JACK, WASAPI, Core Audio, ...) are loaded at run time, so
 * the binary gains no link-time audio dependency. */
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#include <miniaudio.h>
