// Host unit test for the GGML_HEXAGON_VCORNER parser (htp-vcorner.h).
// Builds with the plain host C compiler; see CMakeLists.txt next to it.
#include "htp-vcorner.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK_NAME(str, expected)                                                              \
    do {                                                                                       \
        int got = ggml_hexagon_vcorner_from_name(str);                                         \
        if (got != (expected)) {                                                               \
            printf("FAIL %s:%d: vcorner_from_name(\"%s\") = %d, want %d\n",                    \
                    __FILE__, __LINE__, str ? str : "(null)", got, (expected));                \
            failures++;                                                                        \
        }                                                                                      \
    } while (0)

#define CHECK_SPEC(spec, want_core, want_bus)                                                  \
    do {                                                                                       \
        int core = -12345, bus = -12345, rc = ggml_hexagon_vcorner_spec_parse(spec, &core, &bus); \
        if (rc != 0 || core != (want_core) || bus != (want_bus)) {                             \
            printf("FAIL %s:%d: spec_parse(\"%s\") = rc %d core %d bus %d, want rc 0 core %d bus %d\n", \
                    __FILE__, __LINE__, spec ? spec : "(null)", rc, core, bus,                 \
                    (want_core), (want_bus));                                                  \
            failures++;                                                                        \
        }                                                                                      \
    } while (0)

#define CHECK_SPEC_BAD(spec)                                                                   \
    do {                                                                                       \
        int core = -12345, bus = -12345, rc = ggml_hexagon_vcorner_spec_parse(spec, &core, &bus); \
        if (rc != -1) {                                                                        \
            printf("FAIL %s:%d: spec_parse(\"%s\") = rc %d core %d bus %d, want rc -1\n",      \
                    __FILE__, __LINE__, spec ? spec : "(null)", rc, core, bus);                \
            failures++;                                                                        \
        }                                                                                      \
    } while (0)

static void test_names(void) {
    CHECK_NAME("MAX",        255);
    CHECK_NAME("TURBO_PLUS",   7);
    CHECK_NAME("TURBO",        6);
    CHECK_NAME("NOM_PLUS",     5);
    CHECK_NAME("NOM",          4);
    CHECK_NAME("SVS_PLUS",     3);
    CHECK_NAME("SVS",          2);
    CHECK_NAME("SVS2",         1);
}

static void test_raw_numbers(void) {
    CHECK_NAME("1",     1);
    CHECK_NAME("11",    11);  // TURBO_L5, top of the contiguous range
    CHECK_NAME("255",   255); // MAX
    CHECK_NAME("010",   10);  // base 10 on purpose: not octal
}

static void test_bad_inputs(void) {
    CHECK_NAME(NULL,   -1);
    CHECK_NAME("",     -1);
    CHECK_NAME("bogus",    -1);
    CHECK_NAME("max",      -1);  // case-sensitive
    CHECK_NAME("TURBO ",   -1);
    CHECK_NAME("0x93",     -1);
    CHECK_NAME("0",        -1);  // DISABLE is not a corner vote
    CHECK_NAME("12",       -1);  // hole above TURBO_L5
    CHECK_NAME("254",      -1);  // hole below MAX
    CHECK_NAME("256",      -1);
    CHECK_NAME("-1",       -1);
    CHECK_NAME("1,2",      -1);  // comma belongs to the spec parser
    CHECK_NAME("5x",       -1);
}

static void test_specs(void) {
    CHECK_SPEC("TURBO",            6,    6);
    CHECK_SPEC("TURBO,NOM",        6,    4);
    CHECK_SPEC("SVS2,NOM_PLUS",    1,    5);
    CHECK_SPEC("3,2",              3,    2);
    CHECK_SPEC("255",              255,  255);
    CHECK_SPEC("NOM,255",          4,    255);
}

static void test_bad_specs(void) {
    CHECK_SPEC_BAD(NULL);
    CHECK_SPEC_BAD("");
    CHECK_SPEC_BAD("NOM,");
    CHECK_SPEC_BAD(",NOM");
    CHECK_SPEC_BAD(",");
    CHECK_SPEC_BAD("NOM,NOM,NOM");
    CHECK_SPEC_BAD("NOM,bogus");
    CHECK_SPEC_BAD(" NOM");
    CHECK_SPEC_BAD("256");
    CHECK_SPEC_BAD("TURBO,256");
    char long_spec[128];
    memset(long_spec, 'A', sizeof(long_spec) - 1);
    long_spec[sizeof(long_spec) - 1] = '\0';
    CHECK_SPEC_BAD(long_spec);
}

int main(void) {
    test_names();
    test_raw_numbers();
    test_bad_inputs();
    test_specs();
    test_bad_specs();

    if (failures) {
        printf("htp-vcorner: %d failure(s)\n", failures);
        return 1;
    }
    printf("htp-vcorner: all parser checks passed\n");
    return 0;
}
