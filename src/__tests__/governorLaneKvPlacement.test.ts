/**
 * Source pin: the resolved HTP lane keeps the KV cache ON the device and
 * makes both governor contexts agree on its type. The engine sizes the KV
 * buffers from the layer device (llama-kv-cache.cpp), so prefill KV lands
 * in HTP rpcmem — host-addressable, so the prefill->decode commit stays a
 * host memcpy. HTP writes KV rows only as F32/F16/Q8_0 (SET_ROWS
 * destinations, ggml_hexagon_supported_set_rows; FLASH_ATTN_EXT reads K/V
 * as F16/Q8_0), and the commit copies rows byte-for-byte with
 * llama_kv_commit_access::compatible refusing mismatched cache types — so
 * an HTP-unwritable caller type (the catalog ships V q4_0) is upgraded to
 * q8_0 in BOTH params, once per load, on KALSA_KV_TYPE_OVERRIDE. Measured
 * (lab 2026-10-01/02): KV on HTP is 6.6-8.8x prefill at 3.1k context and V
 * q8_0 is also the better quant (mean KLD vs f16 0.0091 -> 0.0025); cost
 * +2,048 B/token per cache.
 */
import { readFileSync } from "fs";
import { join } from "path";

const source = readFileSync(join(__dirname, "../../cpp/rn-llama.cpp"), "utf8");

/**
 * Body of the block opened at `marker` (the first `{` at or after it),
 * found by brace matching so the slice cannot cross the block's closing
 * brace the way a flat `.*` regex would. Nothing in the matched region
 * has braces inside comments or string literals.
 */
function blockAfter(text: string, marker: string): string | null {
  const at = text.indexOf(marker);
  if (at === -1) return null;
  const open = text.indexOf("{", at);
  if (open === -1) return null;
  let depth = 0;
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(open + 1, i);
    }
  }
  return null;
}

describe("governor lane KV placement pin", () => {
  const fn = blockAfter(source, "bool load_governor_models(");
  const useDevice = blockAfter(fn ?? "", "if (device_plan.use_device) {");
  const count = (text: string | null, needle: string) =>
    text === null ? 0 : text.split(needle).length - 1;

  test("load_governor_models still exists in the expected shape", () => {
    expect(fn).not.toBeNull();
    expect(useDevice).not.toBeNull();
  });

  test("the resolved HTP lane no longer pins the KV to host memory", () => {
    // The device holds the KV: no assignment to no_kv_offload may exist
    // anywhere in load_governor_models (the publish reads the flag to name
    // the placement — reads are fine, assignments are the pin).
    expect(fn).toMatch(/\.no_kv_offload\s*\?/);
    expect(fn).not.toMatch(/\.no_kv_offload\s*=[^=]/);
  });

  test("an HTP-unwritable caller type is upgraded to q8_0 in BOTH contexts", () => {
    // The commit copies rows byte-for-byte and compatible() refuses
    // mismatched cache types, so each upgrade must land on prefill_params
    // and decode_params together, exactly once per side.
    (["cache_type_k", "cache_type_v"] as const).forEach((side) => {
      expect(count(useDevice, `prefill_params.${side} = GGML_TYPE_Q8_0;`)).toBe(1);
      expect(count(useDevice, `decode_params.${side} = GGML_TYPE_Q8_0;`)).toBe(1);
    });
    // Nothing outside the use_device block assigns cache types: a
    // GPU-degraded lane and the lane-off load keep the caller's catalog
    // types byte for byte.
    expect(count(fn, "GGML_TYPE_Q8_0;")).toBe(4);
  });

  test("the upgrade is gated on the HTP-writable set, not on a hardcoded type", () => {
    expect(source).toMatch(
      /bool htp_kv_type_supported\(ggml_type type\) {\s*return type == GGML_TYPE_F32 \|\| type == GGML_TYPE_F16 \|\| type == GGML_TYPE_Q8_0;/,
    );
    expect(count(useDevice, "!htp_kv_type_supported(prefill_params.cache_type_k)")).toBe(1);
    expect(count(useDevice, "!htp_kv_type_supported(prefill_params.cache_type_v)")).toBe(1);
  });

  test("an upgrade is announced once per load on KALSA_KV_TYPE_OVERRIDE", () => {
    // The catalog is the authority; overriding it for this lane must be
    // visible in the load log.
    expect(count(useDevice, "KALSA_KV_TYPE_OVERRIDE {requested_k:")).toBe(1);
    expect(count(fn, "KALSA_KV_TYPE_OVERRIDE {requested_k:")).toBe(1);
  });

  test("the load publishes the effective KV facts for the stats surface", () => {
    expect(count(fn, "owner.setGovernorKvCache(")).toBe(1);
  });

  test("the lane-off branch places no KV decision of its own", () => {
    // Lane off reproduces the CPU/OpenCL-only load byte for byte; its KV
    // placement and types are the engine defaults and must not drift.
    const laneOff = source.match(
      /Lane off: what the CPU\/OpenCL-only build did[\S\s]*?decode_params\.devices\.push_back\(nullptr\);/,
    );
    expect(laneOff).not.toBeNull();
    expect(laneOff![0]).not.toContain("no_kv_offload");
    expect(laneOff![0]).not.toContain("cache_type");
  });
});
