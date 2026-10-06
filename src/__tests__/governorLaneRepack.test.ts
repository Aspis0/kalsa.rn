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
    // Window: function start to the prefill init. Was 2400 and went stale
    // when the lane device-resolution block grew the function past it, then
    // 6000 and went stale again when the lane's KV type-upgrade block grew
    // it further, then 9000 and went stale when the one-copy load block
    // (rn-legs-table branch + load_governor_one_model) landed between the
    // function head and the prefill init.
    const lane = source.match(
      /bool load_governor_models\([\S\s]{0,16000}?common_init_from_params\(prefill_params/,
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

  test("the one-copy loader is the only other no_extra_bufts site, on its own local params", () => {
    // Two sites total: the conditional decode_params flip pinned above, and
    // the one-copy loader's unconditional true on ITS OWN local copy of
    // owner.params (FABLE (b): one model, no CPU_REPACK second copy). The
    // owner's repack policy is still only flipped by the conditional site.
    const assignments = source.match(/\.no_extra_bufts\s*=/g) ?? [];
    expect(assignments).toHaveLength(2);
    const onecopySite = source.match(/[^\w.]params\.no_extra_bufts = true;/);
    expect(onecopySite).not.toBeNull();
  });
});
