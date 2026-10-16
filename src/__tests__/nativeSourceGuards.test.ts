/**
 * Source-text guards for binding-native invariants jest cannot execute: the
 * governor failure error prefix is a stable API (the JSI throw is the delivery
 * channel to JS), the eval-failure log must not carry prompt text into
 * logcat, and logToJsCallback must not forward DEBUG lines, which carry the
 * same prompt text.
 */
import fs from 'fs'
import path from 'path'

const cpp = (name: string) =>
  fs.readFileSync(path.join(__dirname, '../../cpp', name), 'utf8')

test('governor decode failure error prefix is stable', () => {
  // The app recognizes a decode-time governor failure by this prefix of the
  // thrown error (RNLlamaJSI.cpp:1309-1313 runs before the result is
  // serialized, so the throw - not any result field - is the channel).
  expect(cpp('jsi/RNLlamaJSI.cpp')).toContain('"Governor decode failed: "')
})

const contentMacroRegions = (source: string): string[] =>
  source.match(/#ifdef RNLLAMA_LOG_CONTENT[\S\s]*?#endif/g) ?? []

// True when the needle exists nowhere outside a content-logging region.
const gatedContent = (source: string, needle: string) =>
  !contentMacroRegions(source)
    .reduce((outside, region) => outside.replace(region, ''), source)
    .includes(needle)

test('content log formats sit only inside the content macro', () => {
  // Each needle must exist (deleting the format would silently pass the gate
  // check) and appear only inside #ifdef RNLLAMA_LOG_CONTENT regions, so the
  // plain release APK never compiles them.
  const needles: Array<[string, string]> = [
    ['rn-mtmd.hpp', 'content=%s'],
    ['rn-mtmd.hpp', 'prompt: %s'],
    ['rn-completion.cpp', 'cache_head=[%s]'],
    ['rn-completion.cpp', 'shared_txt=[%s]'],
    ['rn-completion.cpp', "text='%s'"],
    ['rn-slot-manager.cpp', "Stopped on word '%s'"],
  ]
  const results = needles.map(([file, needle]) => {
    const source = cpp(file)
    return { file, needle, present: source.includes(needle), gated: gatedContent(source, needle) }
  })
  expect(results).toEqual(
    needles.map(([file, needle]) => ({ file, needle, present: true, gated: true })),
  )
})

test('content logging Gradle flag is always explicit', () => {
  // Both branches must pass the CMake variable, so a reused cache can never
  // keep content logging ON after the property is removed.
  const gradle = fs.readFileSync(
    path.join(__dirname, '../../android/build.gradle'),
    'utf8',
  )
  expect(gradle).toMatch(/-DRNLLAMA_LOG_CONTENT=\${rnllamaLogContent \? "ON" : "OFF"}/)
})

test('KVDIAG0 counts stay unconditional', () => {
  // Release builds keep the counts-only signal; the ids live in the macro.
  expect(cpp('rn-completion.cpp')).toContain('KALSA_KVDIAG0 cache_len=%zu prompt_len=%zu"')
})

test('eval-failure log carries no token text', () => {
  // Anchored on ', n_eval' so it cannot match the unrelated
  // 'failed to eval speaker prefix' statement earlier in the file.
  const logStatement = cpp('rn-completion.cpp').match(
    /LOG_ERROR\("failed to eval, n_eval[\S\s]*?\);/,
  )
  expect(logStatement).not.toBeNull()
  expect(logStatement![0]).not.toContain('tokens_to_str')
})

test('logToJsCallback never forwards DEBUG', () => {
  // Pins two shapes, not the logic: DEBUG is never named in the forwardable
  // set, and the early return on a non-forwardable line comes before the
  // stderr write. The INFO needle keeps a deleted set from passing silently.
  const fn = cpp('jsi/RNLlamaJSI.cpp').match(
    /static void logToJsCallback\([\S\s]*?^ {4}}\n/m,
  )
  expect(fn).not.toBeNull()
  expect(fn![0]).toContain('GGML_LOG_LEVEL_INFO')
  expect(fn![0]).not.toContain('GGML_LOG_LEVEL_DEBUG')
  const gate = fn![0].indexOf('if (!forwardable)')
  expect(gate).toBeGreaterThan(-1)
  expect(gate).toBeLessThan(fn![0].indexOf('llama_log_callback_default'))
})

// Every assignment to a common_params devices list in the loader file. The
// vector is handed to the engine as a raw array and the walk stops at the
// first nullptr (vendor/llama.cpp/src/llama.cpp llama_prepare_model_devices);
// common_model_params_to_llama only skips an EMPTY list, so any non-empty
// list built here must carry its own terminator.
const devicesListSites = (
  source: string,
): Array<{ site: string; terminated: boolean }> => {
  const sites: Array<{ site: string; terminated: boolean }> = []
  const assignment = /(\w+)\.devices\s*=\s*([^;]+);/g
  for (let hit = assignment.exec(source); hit !== null; hit = assignment.exec(source)) {
    const statement = hit[0]!
    const target = hit[1]!
    const rhs = hit[2]!.trim()
    if (rhs.startsWith('{')) {
      // Brace initializer: nullptr must be the last element.
      sites.push({ site: statement, terminated: /nullptr\s*}$/.test(rhs) })
    } else if (rhs.includes('.devices')) {
      // Copy of another devices list: it inherits that list's state (the
      // draft list is filled only by upstream's nullptr-terminated
      // parse_device_list, or left empty, which the loader skips), so a
      // pushed terminator here would turn "default device selection" into
      // an explicit no-offload list.
      sites.push({ site: statement, terminated: true })
    } else {
      // A fresh vector must push the terminator as the very next statement.
      const after = source.slice(hit.index + statement.length).trimStart()
      sites.push({
        site: statement,
        terminated: after.startsWith(`${target}.devices.push_back(nullptr)`),
      })
    }
  }
  return sites
}

test('every devices list in rn-llama.cpp is nullptr-terminated', () => {
  // Non-empty lists are walked element by element with no bounds check, so a
  // missing terminator reads past the vector (one-copy load SIGSEGV).
  const sites = devicesListSites(cpp('rn-llama.cpp'))
  expect(sites.length).toBeGreaterThan(0)
  expect(sites.filter((site) => !site.terminated)).toEqual([])
})

test('bench path drops the observation record only, never engine state', () => {
  // reset_prefill_stats resets engine routing (hop clock, route, legs,
  // prefill stats), so the bench - which restarts its sequence per run -
  // may drop the stage 0 record but must not touch the engine. Pin the
  // whole bench function: exactly the two record resets, no engine reset
  // name anywhere in it.
  const bench = cpp('rn-completion.cpp').match(
    /json llama_rn_context_completion::bench\(int pp, int tg, int pl, int nr\) {[\S\s]*?\n}\n/,
  )![0]
  expect(bench.match(/->reset_observation_record\(\)/g)?.length).toBe(2)
  expect(bench).not.toContain('resetGovernorPrefillStats')
  expect(bench).not.toContain('llama_governor_reset_prefill_stats')
  // And the public record reset stays engine-free: the engine prefill-stats
  // reset exists exactly once in rn-governor.cpp, inside reset_prefill_stats.
  expect(cpp('rn-governor.cpp').split('llama_governor_reset_prefill_stats(').length - 1).toBe(1)
})

test('governor stats wire decode_tokens_gpu beside the cpu/npu counters', () => {
  // The KALSA_GOVERNOR GPU leg evidence is this property; a dropped or
  // relocated setProperty still passes tsc, so pin the adjacency on the wire.
  expect(cpp('jsi/RNLlamaJSI.cpp')).toMatch(
    /result\.setProperty\(rt, "decode_tokens_cpu"[^\n]*\n[^\n]*"decode_tokens_npu"[^\n]*\n[^\n]*"decode_tokens_gpu"/,
  )
})

test('governor stats wire decode_us_* adjacency', () => {
  // Stage 0's per-leg decode time is the tok/s evidence (tok x 1e6 / us);
  // pin the three fields' adjacency on the wire, same shape as the
  // decode_tokens_* guard above.
  expect(cpp('jsi/RNLlamaJSI.cpp')).toMatch(
    /result\.setProperty\(rt, "decode_us_cpu"[^\n]*\n[^\n]*"decode_us_npu"[^\n]*\n[^\n]*"decode_us_gpu"/,
  )
})

test('the turn\'s prefill placement rides the stats wire', () => {
  // last_router_rule cannot answer this: every decode token overwrites it, so
  // the turn's own rule and the margins its route was taken on have to come
  // from the fields the engine keeps per turn. A dropped or relocated
  // setProperty still passes tsc, so pin the four together, in the prefill
  // block they belong to.
  const jsi = cpp('jsi/RNLlamaJSI.cpp')
  expect(jsi).toMatch(
    /"prefill_ctx_ngl"[^\n]*\n[\S\s]*?"prefill_router_rule", \(double\) stats\.prefill_router_rule[\S\s]*?"prefill_thermal_handoff", \(bool\) stats\.prefill_thermal_handoff[\S\s]*?"prefill_npu_headroom_c"[\S\s]*?"prefill_gpu_headroom_c"/,
  )
  // The rule is a count, the handoff a flag: neither may cross as a string.
  expect(jsi).toContain('"prefill_thermal_handoff", (bool) stats.prefill_thermal_handoff')
  // NaN has no JSON spelling, so a turn that placed no prefill (or one placed
  // on no sample) must read as null, never as a number - the same rule the
  // decode_pacing_headroom_last field above already follows.
  ;['npu', 'gpu'].forEach((leg) => {
    expect(jsi).toMatch(
      new RegExp(
        `"prefill_${leg}_headroom_c",\\s*\\n?[\\S\\s]{0,120}?std::isfinite\\(stats\\.prefill_${leg}_headroom_c\\)` +
          `[\\S\\s]{0,120}?jsi::Value\\(\\(double\\) stats\\.prefill_${leg}_headroom_c\\)` +
          `[\\S\\s]{0,40}?: jsi::Value::null\\(\\)`,
      ),
    )
  })

  // The TS type documents the rule numbers for the app author, so the engine's
  // own list (the comment over prefill_rule_for) decides which numbers that
  // documentation may name, and whether each is a prefill or a decode stamp.
  // Word-bounded, so "1" cannot be satisfied by any other digit in the file.
  const engine = fs.readFileSync(
    path.join(__dirname, '../../vendor/llama.cpp/src/llama-governor-policy.cpp'),
    'utf8',
  )
  const ruleKind = (rule: string): string =>
    engine.match(new RegExp(`^//\\s+${rule}\\s+(\\w+)`, 'm'))?.[1] ?? ''
  const doc = fs.readFileSync(path.join(__dirname, '../../src/types.ts'), 'utf8').match(
    /\/\*\* Which rule placed THIS turn's prefill[\S\s]*?\*\//,
  )![0]
  const prefillRules = ['1', '2', '5', '8', '9', '10']
  const decodeRules = ['3', '4']
  // Each named number must exist in the engine's list under the same kind the
  // TS doc claims, so a renumbering upstream cannot leave the docs behind.
  expect(prefillRules.map(ruleKind)).toEqual(prefillRules.map(() => 'prefill'))
  expect(decodeRules.map(ruleKind)).toEqual(decodeRules.map(() => 'decode'))
  // ...and the doc itself must still name all of them.
  expect(
    [...prefillRules, ...decodeRules].filter((rule) => !new RegExp(`\\b${rule}\\b`).test(doc)),
  ).toEqual([])
  expect(doc).toContain('3 and 4 are decode-only')
})

test('the load hands JS a governor_load record, or none at all', () => {
  // The uploaded report reads the JS console, so the load facts (which legs the
  // engine built, the effective KV types, the matched row, the zone counts)
  // reach the app only through this field. Pin it on the init result and the
  // mapping it calls.
  expect(cpp('jsi/RNLlamaJSI.cpp')).toContain(
    '{"governor_load", rnllama::governor_load_json(ctx->governorLoad())}',
  )
  expect(cpp('rn-governor-params.cpp')).toContain(
    'nlohmann::ordered_json governor_load_json(const governor_load_facts & facts)',
  )
  // A load that published nothing crosses as JSON null, never as an empty
  // record: an empty one would read as a governor load that built no legs.
  expect(cpp('rn-governor-params.cpp')).toMatch(
    /if \(!facts\.governor\) {\n {8}return nullptr;\n {4}}/,
  )
  // Both governor success paths must publish, or a load loses its record
  // silently while the field stays on the wire.
  const loader = cpp('rn-llama.cpp')
  expect(loader.split('publish_governor_load(').length - 1).toBe(3)
  // The one-copy legs come from the ENGINE's probe: only it knows which legs it
  // refused (a dropped leg is logged as onecopy_leg_absent and reads false).
  expect(loader).toMatch(
    /case onecopy_load_result::loaded: {[\S\s]*?legs\.npu = owner\.governor->legBuilt\(llama_governor_engine::NPU\);[\S\s]*?legs\.gpu = owner\.governor->legBuilt\(llama_governor_engine::GPU\);[\S\s]*?publish_governor_load\(\n {20}owner, legs, true, hw_facts\.soc_model,/,
  )
  // The two-model legs come from the placement THIS load applied, never from
  // the engine: it builds contexts rather than legs, so legBuilt would report
  // no accelerator on a load that ran one. They go through the pure resolver,
  // which is where the HTP-pinned case is handled: a resolved lane puts
  // ctx_prefill on the HTP alone ({resolved.device, nullptr}), so prefill_cpu_only
  // is FALSE there and reading the GPU leg off it claims a leg that load never
  // placed. The helper exists precisely to keep that trap out of the loader.
  expect(loader).toMatch(
    /publish_governor_load\(\n {8}owner, governor_two_model_legs\(policy_params\.npu_lane_enabled, gpu_allowed\),/,
  )
  expect(loader).not.toMatch(/legs\.gpu = !prefill_cpu_only;/)
  expect(cpp('rn-governor-params.cpp')).toMatch(
    /legs\.npu = lane_resolved;[\S\s]{0,240}?legs\.gpu = gpu_placement_allowed && !lane_resolved;/,
  )
  // The row's SoC rides the two-model path too: only one_copy=false tells a
  // demotion apart from a load that matched nothing.
  expect(loader).toMatch(
    /governor_two_model_legs\([^\n]*\),\n {8}false, leg_set\.one_copy \? hw_facts\.soc_model : std::string\(\),/,
  )
})

test('the governor_load keys agree across the wire, the types and the d.ts', () => {
  // The mapping, the TS type and the generated declarations are three files
  // that must name the same keys: a rename on one side silently drops the fact
  // from the uploaded report. Parse each rather than substring-match, so a key
  // that only appears in a comment cannot satisfy the check.
  const cppKeys = Array.from(
    cpp('rn-governor-params.cpp').match(
      /return nlohmann::ordered_json{([\S\s]*?)};/,
    )![1]!.matchAll(/{"([_a-z]+)",/g),
  ).map((m) => m[1]!)

  const tsKeys = (source: string): string[] => {
    const body = source.match(/export type GovernorLoadFacts = {([\S\s]*?)\n}/)![1]!
    // Indent-agnostic: tsc re-emits the declared file with four spaces.
    return Array.from(body.matchAll(/^\s+([_a-z]+)[:?]/gm)).map((m) => m[1]!)
  }

  expect(cppKeys.length).toBeGreaterThan(0)
  const source = fs.readFileSync(path.join(__dirname, '../../src/types.ts'), 'utf8')
  expect(tsKeys(source)).toEqual(cppKeys)
  expect(tsKeys(fs.readFileSync(path.join(__dirname, '../../lib/typescript/types.d.ts'), 'utf8')))
    .toEqual(cppKeys)

  // And the init result types the record as always present but nullable: the
  // C++ emits the key on every load and JSON null on a non-governor one, so an
  // optional field would let a reader believe the key itself can be missing.
  expect(source).toMatch(/systemInfo: string\n {2}\/\*\*[\S\s]*?\*\/\n {2}governor_load: GovernorLoadFacts \| null\n}/)
})

// Body of one member function of rn-governor.cpp (its closing brace is the
// first at column 0).
const governorFn = (signature: RegExp): string =>
  cpp('rn-governor.cpp').match(new RegExp(signature.source + /[^\n]*[\S\s]*?\n}\n/.source))![0]

test('the prefill thermal reader is built without the hop', () => {
  // With decode_hop_tokens at its production default of 0 nothing read a zone
  // at all, so the prefill placement had no margins to price. The reader is
  // therefore built on its own account once a one-copy load has both legs it
  // compares - and the hop's cadence (decode_hop_tokens_, which still gates
  // every decode-path sample) stays a separate fact. rn_governor needs a model
  // to construct, so the host tests cannot reach this; the structure is the
  // contract.
  const init = governorFn(/void rn_governor::init_hop_reader\(const llama_governor_params & params, bool prefill_thermal\)/)
  expect(init).toMatch(
    /if \(params\.decode_hop_tokens > 0\) {[\S\s]*?} else if \(!prefill_thermal\) {\n {8}return;\n {4}}\n {4}decode_legs_ = std::make_unique<rn_thermal_legs>/,
  )
  // The 1-token path must not divide by a cadence that was never configured.
  const before = governorFn(/void rn_governor::before_decode_batch\(int n_tokens\)/)
  expect(before).toMatch(
    /if \(decode_hop_tokens_ == 0\) {[\S\s]*?return;\n {4}}\n {4}const int64_t now_ms/,
  )
  // One-copy asks for the reader with the legs the engine actually built - a
  // demoted leg is one the handoff could never reach - and the two-model form
  // has no GPU leg to price, so it passes false.
  const constructors = cpp('rn-governor.cpp')
  expect(constructors.split('init_hop_reader(params, npu_leg_built && gpu_leg_built)').length - 1).toBe(1)
  expect(constructors.split('init_hop_reader(params, false)').length - 1).toBe(1)
  expect(constructors).toMatch(
    /llama_governor_leg_ctx\(governor_, legs\[i]\.engine\) == nullptr\) {[\S\s]*?continue;\n {8}}[\S\s]*?init_hop_reader\(params, npu_leg_built && gpu_leg_built\)/,
  )
})

test('the prefill margins are pushed once per turn, before admission', () => {
  const before = governorFn(/void rn_governor::before_decode_batch\(int n_tokens\)/)
  // The push lives in the multi-token branch only: a decode batch never
  // re-prices the turn, and the latch makes the turn's first prefill batch the
  // only one that reads the zones (a chunked prefill is several batches).
  const prefillBranch = before.slice(before.indexOf('if (n_tokens > 1)'))
  expect(prefillBranch).toMatch(
    /if \(!prefill_headroom_sampled_\) {\n {12}prefill_headroom_sampled_ = true;\n {12}push_prefill_headroom\(\);\n {8}}/,
  )
  expect(before.indexOf('push_prefill_headroom()')).toBeLessThan(
    before.indexOf('if (decode_hop_tokens_ == 0)'),
  )
  // The engine drops its copy wherever the route re-arms, so the latch does too:
  // a turn the binding does not sample must read as unknown, not as stale heat.
  expect(
    ['clear_cache', 'reset_prefill_stats'].filter(
      (fn) => !governorFn(new RegExp(`void rn_governor::${fn}\\(`)).includes('prefill_headroom_sampled_ = false'),
    ),
  ).toEqual([])
  // And the push itself is one engine call plus the per-turn line.
  const push = governorFn(/void rn_governor::push_prefill_headroom\(\)/)
  expect(push).toContain('llama_governor_set_prefill_headroom(governor_, legs.npu_headroom_c, legs.gpu_headroom_c)')
  expect(push).toContain('KALSA_GOVERNOR_PREFILL_THERMAL')
})

test('the per-turn thermal line parses as JSON, unknown leg as null', () => {
  // Same reason as the battery line: the campaign parser drops a payload it
  // cannot JSON.parse. The two margins are substituted, so this fails if the
  // format ever carries a bare nan.
  const push = governorFn(/void rn_governor::push_prefill_headroom\(\)/)
  const format = push
    .match(/"((?:[^"\\]|\\.)*)"/g)!
    .map((literal) => literal.slice(1, -1))
    .join('')
  const tag = 'KALSA_GOVERNOR_PREFILL_THERMAL '
  const payload = format.slice(format.indexOf(tag) + tag.length).replace(/\\"/g, '"')
  expect(JSON.parse(payload.replace(/%s/g, 'null'))).toEqual({
    npu_headroom_c: null,
    gpu_headroom_c: null,
  })
  expect(JSON.parse(payload.replace(/%s/g, '7.5'))).toEqual({
    npu_headroom_c: 7.5,
    gpu_headroom_c: 7.5,
  })
  // And the formatter is what turns an unreadable leg into that null: never 0,
  // never a printed nan.
  const formatter = cpp('rn-governor.cpp').match(/std::string json_number\(float value\) {[\S\s]*?\n}\n/)![0]
  expect(formatter).toContain('std::isfinite(value) ? std::to_string(value) : "null"')
})

test('window observation stays fresh on unchanged forecasts', () => {
  // A stable forecast is re-read every 2 s; the observation stamp must move
  // on every finite read even when the unchanged-value early return skips
  // the engine update, or the logged age grows without bound (hostile audit
  // item 1). Pin the store ahead of that early return, inside the read.
  const refresh = governorFn(/void rn_governor::refresh_platform_thermal\(\)/)
  const read = refresh.indexOf('if (headroom_due)')
  const stamp = refresh.indexOf('observed_headroom_read_us_ = platform_headroom_state_.read_us')
  const earlyReturn = refresh.indexOf('if (!decision.send && !headroom_changed)')
  expect(read).toBeGreaterThan(-1)
  expect(stamp).toBeGreaterThan(read)
  expect(earlyReturn).toBeGreaterThan(stamp)
})

test('KALSA_GOVERNOR_BATTERY_SYSFS line parses as JSON', () => {
  // The campaign parser JSON.parses the payload verbatim and silently drops
  // what does not parse (telemetryParse.mjs), so the emitted line must be
  // strict JSON: quoted keys, numeric values. Rebuild the payload from the
  // LOG_INFO format string (adjacent C++ literals joined) and parse it here.
  const fn = cpp('rn-governor.cpp').match(/void rn_governor::log_battery_sysfs[\S\s]*?\n}\n/)
  expect(fn).not.toBeNull()
  const format = fn![0]
    .match(/"((?:[^"\\]|\\.)*)"/g)!
    .map((literal) => literal.slice(1, -1))
    .join('')
  const tag = 'KALSA_GOVERNOR_BATTERY_SYSFS '
  const payload = format
    .slice(format.indexOf(tag) + tag.length)
    .replace(/\\"/g, '"')
    .replace(/%d/g, '1')
  expect(JSON.parse(payload)).toEqual({
    temp_ok: 1,
    temp_tenths_c: 1,
    status_ok: 1,
    status: 1,
  })
})

test('decode-path observation hooks stay wired', () => {
  // The host tests drive the record's own API, so nothing fails if the
  // governor stops calling it. Pin the whole chain: both completion loops
  // reach before_decode_batch, that stages the window (no record mutator
  // runs there - a paused first batch must leave the record empty, the
  // commit happens only past the rc 0 gate in decode), commit_window is the
  // one place the record mutators live, and gpu-coolmode carries its own
  // leg marker instead of reading as unknown.
  const calls = cpp('rn-completion.cpp').match(/->before_decode_batch\(/g)
  expect(calls?.length).toBe(2)
  const before = governorFn(/void rn_governor::before_decode_batch\(int n_tokens\)/)
  expect(before).toContain('stage_window(now_ms, legs)')
  const recordMutators = [
    'samples_.record(',
    'samples_.first_step(',
    'samples_.last_step(',
  ]
  expect(recordMutators.filter((mutator) => before.includes(mutator))).toEqual([])
  const governorCpp = cpp('rn-governor.cpp')
  // Every record mutator call exists EXACTLY ONCE in rn-governor.cpp - the
  // class definitions in rn-governor-samples.h aside, commit_window owns
  // these three and reset_observation_record owns the reset - so no second
  // site can grow beside them.
  const mutatorCalls = [...recordMutators, 'samples_.reset(']
  expect(
    mutatorCalls.filter((mutator) => governorCpp.split(mutator).length - 1 !== 1),
  ).toEqual([])
  const decode = governorFn(/int32_t rn_governor::decode\(llama_batch batch\)/)
  // The commit is the whole body of the rc-0 branch - nothing commits when
  // the step did not run.
  expect(decode).toMatch(
    /if \(decode_legs_ != nullptr && batch\.n_tokens == 1 && result == 0\) {\n {12}commit_window\(ggml_time_us\(\) \/ 1000\);\n {8}}/,
  )
  const commit = governorFn(/void rn_governor::commit_window\(int64_t step_end_ms\)/)
  expect(recordMutators.filter((mutator) => !commit.includes(mutator))).toEqual([])
  const stage = governorFn(/void rn_governor::stage_window\(int64_t now_ms/)
  expect(stage).toContain('pending_valid_ = true')
  expect(cpp('rn-governor.cpp')).toMatch(
    /case llama_governor_engine::GPU_COOLMODE:\n {12}return 3;/,
  )
})
