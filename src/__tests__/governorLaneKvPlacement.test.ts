/**
 * Source pin: the resolved HTP lane's one KV decision. On the device the
 * KV rides the layer device into HTP rpcmem and an HTP-unwritable caller
 * type (the catalog ships V q4_0) is upgraded to q8_0 in BOTH contexts —
 * the commit copies rows byte-for-byte and
 * llama_kv_commit_access::compatible refuses mismatched cache types. With
 * flash attention EXPLICITLY disabled the lane keeps the pre-lane-KV load
 * instead: the engine refuses a quantized V cache without FA
 * (llama-context.cpp "quantized V cache requires flash_attn to be
 * enabled"; AUTO self-enables) and stores the V rows transposed
 * (llama-model.cpp attn_v_trans = !cparams.flash_attn), a layout the HTP
 * path was never validated for — so host-pinned KV, caller types,
 * announced once per load on KALSA_KV_HOST_PIN. Measured (lab
 * 2026-10-01/02): KV on HTP is 6.6-8.8x prefill at 3.1k context and V
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
  const decision = blockAfter(source, "governor_lane_kv_plan decide_governor_lane_kv(");
  const faOff = blockAfter(useDevice ?? "", "} else {");
  const count = (text: string | null, needle: string) =>
    text === null ? 0 : text.split(needle).length - 1;

  test("load_governor_models still exists in the expected shape", () => {
    expect(fn).not.toBeNull();
    expect(useDevice).not.toBeNull();
    expect(decision).not.toBeNull();
  });

  test("the lane's KV placement and types come from one pure decision", () => {
    // The decision owns the type gate and the FA exception; the load body
    // only consumes it, fed the caller's flash_attn_type as-is.
    expect(count(fn, "decide_governor_lane_kv(")).toBe(1);
    expect(count(fn, "prefill_params.flash_attn_type);")).toBe(1);
  });

  test("flash attention explicitly disabled keeps the pre-lane-KV load", () => {
    // The DISABLED short-circuit is the only route to kv_on_device=false,
    // and it returns the caller's types untouched; only the device branch
    // names an upgrade type.
    expect(count(decision, "LLAMA_FLASH_ATTN_TYPE_DISABLED")).toBe(1);
    expect(decision).toContain("return { false, requested_k, requested_v };");
    expect(count(decision, "GGML_TYPE_Q8_0")).toBe(2);
  });

  test("the flash-attn-off pin is host KV with caller types, logged once", () => {
    expect(faOff).not.toBeNull();
    expect(count(faOff, "prefill_params.no_kv_offload = true;")).toBe(1);
    expect(count(faOff, "cache_type")).toBe(0);
    expect(count(fn, "KALSA_KV_HOST_PIN {requested_k:")).toBe(1);
    expect(count(fn, 'reason:\\"flash-attn-off\\"')).toBe(1);
  });

  test("no_kv_offload is assigned only by the flash-attn-off exception", () => {
    // The device-resident lane must not grow a second host pin: the
    // publish reads the flag to name the placement (reads are fine,
    // assignments are the pin).
    expect(fn).toMatch(/\.no_kv_offload\s*\?/);
    expect(count(fn, "prefill_params.no_kv_offload = true;")).toBe(1);
  });

  test("on the device an HTP-unwritable caller type lands as q8_0 in BOTH contexts", () => {
    // The commit copies rows byte-for-byte and compatible() refuses
    // mismatched cache types, so each upgrade must land on prefill_params
    // and decode_params together, exactly once per side, from the plan.
    (["k", "v"] as const).forEach((side) => {
      expect(count(useDevice, `prefill_params.cache_type_${side} = lane_kv.type_${side};`)).toBe(1);
      expect(count(useDevice, `decode_params.cache_type_${side} = lane_kv.type_${side};`)).toBe(1);
    });
    // No hardcoded type in the load body: the gate lives in the decision.
    expect(count(useDevice, "GGML_TYPE_Q8_0")).toBe(0);
    expect(count(decision, "htp_kv_type_supported(requested_k)")).toBe(1);
    expect(count(decision, "htp_kv_type_supported(requested_v)")).toBe(1);
  });

  test("the decision gates on the HTP-writable set, not on a hardcoded type", () => {
    expect(source).toMatch(
      /bool htp_kv_type_supported\(ggml_type type\) {\s*return type == GGML_TYPE_F32 \|\| type == GGML_TYPE_F16 \|\| type == GGML_TYPE_Q8_0;/,
    );
  });

  test("a device-side override is announced once per load on KALSA_KV_TYPE_OVERRIDE", () => {
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
