/**
 * Source pin: the HTP lane keeps the prefill KV cache out of the HTP
 * buffers. The engine sizes the KV buffers from the layer device
 * (llama-kv-cache.cpp) unless the context loads with no_kv_offload, and an
 * HTP-resident V cache aborts the load: the HTP SET_ROWS only accepts
 * F32/F16/Q8_0 destinations and the app ships a q4_0 V cache, so the
 * scheduler hits a pre-allocated destination no backend can write
 * (ggml-backend.cpp "pre-allocated tensor ... cannot run the operation
 * (SET_ROWS)", S23 2026-09-30). The lane's validated shape (HTP prefill
 * spike) keeps the prefill attention/KV in host memory.
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
  const lane = source.match(
    /bool load_governor_models\([\S\s]{0,6000}?common_init_from_params\(prefill_params/,
  );

  test("load_governor_models still exists in the expected shape", () => {
    expect(lane).not.toBeNull();
  });

  test("the HTP lane turns off KV offload on the prefill params", () => {
    expect(lane).not.toBeNull();
    expect(lane![0]).toContain("prefill_params.no_kv_offload = true;");
  });

  test("the KV pin is scoped to the resolved HTP device, not the whole lane", () => {
    // A lane that degraded to GPU keeps the OpenCL placement the GPU route
    // has always used, so the assignment must sit inside the use_device
    // block that only fires when HTP0 resolved — and nowhere else in
    // load_governor_models, or a GPU-degraded lane would move its whole KV
    // to host memory under a gpu_fit plan computed for device KV.
    const fn = blockAfter(source, "bool load_governor_models(");
    const useDevice = blockAfter(fn ?? "", "if (device_plan.use_device) {");
    const pin = "prefill_params.no_kv_offload = true;";
    expect(fn).not.toBeNull();
    expect(useDevice).not.toBeNull();
    expect(useDevice).toContain(pin);
    expect(fn!.split(pin).length - 1).toBe(1);
  });

  test("the lane-off branch places no no_kv_offload of its own", () => {
    // Lane off reproduces the CPU/OpenCL-only load byte for byte; its KV
    // placement is the engine default and must not drift with this pin.
    const laneOff = source.match(
      /Lane off: what the CPU\/OpenCL-only build did[\S\s]*?decode_params\.devices\.push_back\(nullptr\);/,
    );
    expect(laneOff).not.toBeNull();
    expect(laneOff![0]).not.toContain("no_kv_offload");
  });
});
