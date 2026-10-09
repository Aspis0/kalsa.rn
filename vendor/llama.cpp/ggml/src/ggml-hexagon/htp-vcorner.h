// Parses the GGML_HEXAGON_VCORNER lab spec ("<corner>[,<corner>]") into
// HAP_dcvs_voltage_corner_t ordinals. Host-side only and compiled without the
// Hexagon SDK, so the ordinals below are inlined from HAP_power.h (SDK 6.x);
// htp/main.c static-asserts them against the real enum on the DSP.
#ifndef GGML_HEXAGON_VCORNER_H
#define GGML_HEXAGON_VCORNER_H

#include <stdlib.h>
#include <string.h>

// GGML_HEXAGON_VCORNER name -> HAP_dcvs_voltage_corner_t, -1 if unknown.
// Case-sensitive on purpose: the accepted names are logged on mismatch.
static inline int ggml_hexagon_vcorner_from_name(const char * s) {
    static const struct { const char * name; int corner; } table[] = {
        { "MAX",        255 },
        { "TURBO_PLUS",   7 },
        { "TURBO",        6 },
        { "NOM_PLUS",     5 },
        { "NOM",          4 },
        { "SVS_PLUS",     3 },
        { "SVS",          2 },
        { "SVS2",         1 },
    };

    if (s == NULL || *s == '\0') {
        return -1;
    }

    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        if (strcmp(table[i].name, s) == 0) {
            return table[i].corner;
        }
    }

    // Raw decimal ordinal (base 10 on purpose: "010" must not read as octal).
    char * end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || *end != '\0') {
        return -1;
    }
    // HAP_dcvs_voltage_corner_t membership: SVS2..TURBO_L5 are 1..11 and MAX is 255;
    // 0 is DISABLE (not a vote) and 12..254 are holes in the enum.
    if (!((v >= 1ul && v <= 11ul) || v == 255ul)) {
        return -1;
    }
    return (int) v;
}

// "<corner>[,<corner>]" -> (core, bus); a missing bus part repeats the core.
// Returns 0 and fills core/bus on success, -1 if any part is unparsable.
static inline int ggml_hexagon_vcorner_spec_parse(const char * spec, int * core, int * bus) {
    if (spec == NULL || core == NULL || bus == NULL) {
        return -1;
    }

    char buf[64];
    size_t len = strlen(spec);
    if (len == 0 || len >= sizeof(buf)) {
        return -1;
    }
    memcpy(buf, spec, len + 1);

    char * bus_part = strchr(buf, ',');
    if (bus_part) {
        *bus_part = '\0';
        bus_part++;
    }

    int c = ggml_hexagon_vcorner_from_name(buf);
    if (c < 0) {
        return -1;
    }
    int b = bus_part ? ggml_hexagon_vcorner_from_name(bus_part) : c;
    if (b < 0) {
        return -1;
    }

    *core = c;
    *bus  = b;
    return 0;
}

#endif /* GGML_HEXAGON_VCORNER_H */
