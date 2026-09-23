/**
 * Source pin: the governor lane's decode model must load with no_extra_bufts
 * so the CPU repack never creates its second full copy of the weights beside
 * the prefill model's OpenCL copy. The behaviour (one copy instead of two on
 * a device) is decided by a phone measurement, not by this pin.
 */
import { readFileSync } from "fs";
import { join } from "path";

const source = readFileSync(join(__dirname, "../../cpp/rn-llama.cpp"), "utf8");

describe("governor lane repack pin", () => {
  test("load_governor_models forces no_extra_bufts on the decode params only", () => {
    const lane = source.match(
      /bool load_governor_models\([\S\s]{0,2400}?common_init_from_params\(prefill_params/,
    );
    expect(lane).not.toBeNull();
    expect(lane![0]).toContain("decode_params.no_extra_bufts = true;");
    // The anon copy comes from the decode model; prefill keeps the owner's
    // params (its tensors live on OpenCL, its CPU residuals have no repack
    // traits — the load log shows no CPU_REPACK line for prefill).
    expect(lane![0]).not.toContain("prefill_params.no_extra_bufts");
  });

  test("no other load path in the binding flips the owner's repack policy", () => {
    const assignments = source.match(/\.no_extra_bufts\s*=/g) ?? [];
    expect(assignments).toHaveLength(1);
  });
});
