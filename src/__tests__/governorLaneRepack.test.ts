/**
 * Source pin: the governor lane's decode model loads with no_extra_bufts only
 * when governor.decode_repack is false (the 8 GB S23 shape, where the lane
 * does not fit with the CPU repack's second copy). Default true keeps
 * upstream repack-on; the 12 GB+ shape where P1 is a pure loss.
 */
import { readFileSync } from "fs";
import { join } from "path";

const source = readFileSync(join(__dirname, "../../cpp/rn-llama.cpp"), "utf8");
const paramsSource = readFileSync(
  join(__dirname, "../../cpp/rn-governor-params.cpp"),
  "utf8",
);

describe("governor lane repack pin", () => {
  test("no_extra_bufts on the decode params is conditional on decode_repack", () => {
    const lane = source.match(
      /bool load_governor_models\([\S\s]{0,2400}?common_init_from_params\(prefill_params/,
    );
    expect(lane).not.toBeNull();
    expect(lane![0]).toContain("if (!load_options.decode_repack) {");
    expect(lane![0]).toContain("decode_params.no_extra_bufts = true;");
    // The anon copy comes from the decode model; prefill keeps the owner's
    // params (its tensors live on OpenCL, its CPU residuals have no repack
    // traits — the load log shows no CPU_REPACK line for prefill).
    expect(lane![0]).not.toContain("prefill_params.no_extra_bufts");
  });

  test("decode_repack parses with an upstream-true default", () => {
    expect(paramsSource).toContain(
      'options.decode_repack = bool_or(governor, "decode_repack", true);',
    );
  });

  test("no other load path in the binding flips the owner's repack policy", () => {
    const assignments = source.match(/\.no_extra_bufts\s*=/g) ?? [];
    expect(assignments).toHaveLength(1);
  });
});
