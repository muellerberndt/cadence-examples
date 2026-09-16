/*
 * The pinned reSIDfp, as the browser calls it.
 *
 * Every call goes through pyreSIDfp::PythonSid, which is the same object the Python binding
 * drives, so the reset order, the mute masking and the clocking are the binding's and not a
 * second implementation of them. This file adds no arithmetic.
 *
 * This file is part of a work derived from pyresidfp and reSIDfp, which are GPL-2.0-or-later.
 */
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <emscripten/emscripten.h>

#include "PythonSid.h"

static pyreSIDfp::PythonSid *chip = nullptr;
static std::string last_error;

extern "C" {

/* The last error any call here caught, as text. Empty while nothing has thrown. */
EMSCRIPTEN_KEEPALIVE const char *sid_last_error(void) { return last_error.c_str(); }

EMSCRIPTEN_KEEPALIVE int sid_create(int model, int method, double clock_hz, double sample_hz) {
    last_error.clear();
    try {
        delete chip;
        chip = nullptr;
        chip = new pyreSIDfp::PythonSid(static_cast<reSIDfp::ChipModel>(model),
                                        static_cast<reSIDfp::SamplingMethod>(method),
                                        clock_hz, sample_hz);
        return 1;
    } catch (const std::exception &error) {
        last_error = error.what();
        return 0;
    } catch (...) {
        last_error = "the chip threw something that is not a std::exception";
        return 0;
    }
}

EMSCRIPTEN_KEEPALIVE void sid_destroy(void) { delete chip; chip = nullptr; }

EMSCRIPTEN_KEEPALIVE void sid_reset(void) { if (chip) chip->reset(); }

EMSCRIPTEN_KEEPALIVE void sid_enable_filter(int on) { if (chip) chip->enableFilter(on != 0); }

EMSCRIPTEN_KEEPALIVE void sid_write(int offset, int value) {
    if (chip) chip->write(offset, static_cast<unsigned char>(value & 0xff));
}

EMSCRIPTEN_KEEPALIVE int sid_read(int offset) { return chip ? static_cast<int>(chip->read(offset)) : 0; }

EMSCRIPTEN_KEEPALIVE void sid_mute(int channel, int on) { if (chip) chip->mute(channel, on != 0); }

EMSCRIPTEN_KEEPALIVE void sid_input(int value) { if (chip) chip->input(value); }

/* Clocks `cycles` chip cycles and copies the samples the chip emitted into `out`, which holds
 * `cap` shorts. Returns how many samples the chip emitted, which may pass `cap`. */
EMSCRIPTEN_KEEPALIVE int sid_clock(unsigned int cycles, short *out, int cap) {
    if (!chip) return 0;
    std::vector<short> samples;
    try {
        samples = chip->clock(cycles);
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
    const int n = static_cast<int>(samples.size());
    const int copied = n < cap ? n : cap;
    for (int i = 0; i < copied; i++) out[i] = samples[static_cast<std::size_t>(i)];
    return n;
}

}
