/**
 * The /bench route surface as CHAIN, not fragments: JS wrapper → JSI host
 * function → context → rn_governor → engine C API, the completion-result
 * emission with its six fields, and the engine's safety-before-override
 * order — each sliced between markers and compared token-for-token
 * (comments stripped), so a dropped hop or a moved gate fails loudly.
 * The TS shape is pinned against the GENERATED declarations, so a stale
 * `lib/typescript` fails too.
 *
 * What stays unpinned without a native build: that the engine actually
 * honours the override at runtime (tests/test-governor-bench-route.cpp is
 * the native harness for that) and that JSI resolves the promise per hop.
 */
import fs from 'fs'
import path from 'path'

const cpp = (name: string) =>
  fs.readFileSync(path.join(__dirname, '../../cpp', name), 'utf8')
const src = (name: string) =>
  fs.readFileSync(path.join(__dirname, '../../src', name), 'utf8')
const generated = (name: string) =>
  fs.readFileSync(path.join(__dirname, '../../lib/typescript', name), 'utf8')
const vendored = (name: string) =>
  fs.readFileSync(path.join(__dirname, '../../vendor/llama.cpp/src', name), 'utf8')

const stripComments = (text: string) =>
  text.replace(/\/\*[\S\s]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')

/** Comment-free, whitespace-collapsed text between two markers, end included. */
function block(source: string, start: string, end: string): string {
  const from = source.indexOf(start)
  if (from < 0) throw new Error(`start marker not found: ${start}`)
  const to = source.indexOf(end, from + start.length)
  if (to < 0) throw new Error(`end marker not found after start: ${end}`)
  return stripComments(source.slice(from, to + end.length)).replace(/\s+/g, ' ').trim()
}

test('the JS wrapper declares the override and binds its native key', () => {
  expect(block(src('index.ts'), 'async setPrefillOverride(', 'await llamaSetPrefillOverride(this.id, mode)')).toBe(
    "async setPrefillOverride(mode: 'cpu' | 'gpu' | 'auto'): Promise<void> { " +
      'const { llamaSetPrefillOverride } = getJsi() ' +
      'await llamaSetPrefillOverride(this.id, mode)',
  )
  expect(src('index.ts')).toMatch(/'llamaSetGovernorThermo',\s*'llamaSetPrefillOverride',/)
  expect(block(src('jsi.ts'), "var llamaSetPrefillOverride:", ') => Promise<boolean>')).toBe(
    "var llamaSetPrefillOverride: ( contextId: number, mode: 'cpu' | 'gpu' | 'auto', ) => Promise<boolean>",
  )
})

test('the JSI hop validates the mode, checks busy and governor, and forwards', () => {
  expect(
    block(
      cpp('jsi/RNLlamaJSI.cpp'),
      '"llamaSetPrefillOverride"),',
      'setProperty(runtime, "llamaSetPrefillOverride", setPrefillOverride);',
    ),
  ).toBe(
    '"llamaSetPrefillOverride"), 2, ' +
      '[callInvoker](jsi::Runtime& runtime, const jsi::Value&, const jsi::Value* arguments, size_t count) -> jsi::Value { ' +
      'if (count < 2 || !arguments[1].isString()) { ' +
      'throw std::invalid_argument("prefill override mode is required"); } ' +
      'const int contextId = (int) arguments[0].asNumber(); ' +
      'const auto mode = arguments[1].asString(runtime).utf8(runtime); ' +
      'if (mode != "auto" && mode != "cpu" && mode != "gpu") { ' +
      'throw std::invalid_argument("prefill override must be cpu, gpu or auto"); } ' +
      'return createPromiseTask(runtime, callInvoker, ' +
      '[contextId, mode]() -> PromiseResultGenerator { ' +
      'auto ctx = getContextOrThrow(contextId); ' +
      'throwIfContextBusy(ctx); ' +
      'if (!ctx->hasGovernor()) { ' +
      'throw std::runtime_error("Governor mode is not enabled"); } ' +
      'const int code = mode == "cpu" ? 1 : mode == "gpu" ? 2 : 0; ' +
      'if (!ctx->setPrefillOverride(code)) { ' +
      'return [](jsi::Runtime&) { return jsi::Value(false); }; } ' +
      'return [](jsi::Runtime&) { return jsi::Value(true); }; ' +
      '}, contextId); } ); ' +
      'runtime.global().setProperty(runtime, "llamaSetPrefillOverride", setPrefillOverride);',
  )
})

test('the context and rn_governor hops forward to the engine call', () => {
  expect(block(cpp('rn-llama.cpp'), 'bool llama_rn_context::setPrefillOverride', '}')).toBe(
    'bool llama_rn_context::setPrefillOverride(int mode) { ' +
      'return governor != nullptr && governor->set_prefill_override(mode); }',
  )
  expect(
    block(
      cpp('rn-governor.cpp'),
      'bool rn_governor::set_prefill_override',
      'return llama_governor_set_prefill_override(governor_, mode);',
    ),
  ).toBe(
    'bool rn_governor::set_prefill_override(int mode) { ' +
      'if (governor_ == nullptr || failed_) { return false; } ' +
      'return llama_governor_set_prefill_override(governor_, mode);',
  )
})

test('the completion result emits route_chunks with the six spec fields', () => {
  const emission = block(
    cpp('jsi/JSICompletion.h'),
    'const auto governor_stats = ctx->governorStats();',
    'res["route_chunks"] = std::move(chunks);',
  )
  expect(emission).toContain('if (ctx->hasGovernor())')
  ;['index', 'requested', 'actual', 'tokens', 'prefill_ms', 'forced'].forEach((field) =>
    expect(emission).toContain(`{"${field}"`),
  )
  expect(emission).toContain('prefillModeName(chunk.requested)')
  expect(emission).toContain('prefillModeName(chunk.actual)')
})

test('the vendored engine declares the API, the facts, and safety before override', () => {
  const ext = vendored('llama-ext.h')
  expect(ext).toContain('LLAMA_API bool llama_governor_set_prefill_override(')
  const struct = block(ext, 'struct llama_governor_route_chunk {', '};')
  ;['index', 'requested', 'actual', 'tokens', 'prefill_ms', 'forced'].forEach((field) =>
    expect(struct).toContain(field),
  )
  // The safety gate must come before the override consult: slice from the
  // function head to the first bench_force reference and compare the order.
  expect(
    block(
      vendored('llama-governor-policy.cpp'),
      'llama_governor_engine llama_governor_policy::prefill_engine() const {',
      'if (params_.bench_force_gpu_prefill',
    ),
  ).toBe(
    'llama_governor_engine llama_governor_policy::prefill_engine() const { ' +
      'if (!valid_schema(params_) || !profile_valid_ || !have_profile_ || ' +
      'state_ == llama_governor_thermal_state::CRITICAL || ' +
      'state_ == llama_governor_thermal_state::Invalid || ' +
      'state_ == llama_governor_thermal_state::LOWBAT) { ' +
      'return llama_governor_engine::CPU; } ' +
      'if (prefill_override_ == llama_governor_prefill_mode::CPU) { ' +
      'return llama_governor_engine::CPU; } ' +
      'if (prefill_override_ == llama_governor_prefill_mode::GPU) { ' +
      'return params_.gpu_fit == llama_governor_fit::Fit ? llama_governor_engine::GPU ' +
      ': llama_governor_engine::CPU; } if (params_.bench_force_gpu_prefill',
  )
  // Facts reset per completion, and every executed prefill records one.
  expect(vendored('llama-governor-runtime.cpp')).toContain('stats_.route_chunk_count = 0;')
  expect(vendored('llama-governor.cpp')).toContain('chunk.forced = mode !=')
})

test('the generated declarations carry the new surface (stale lib fails)', () => {
  expect(generated('index.d.ts')).toContain(
    "setPrefillOverride(mode: 'cpu' | 'gpu' | 'auto'): Promise<void>",
  )
  const pick = (text: string, field: string): string[] => {
    const match = text.match(new RegExp(`${field}\\??:\\s*Array<\\{([^}]*)\\}>`))
    if (match === null) throw new Error(`${field} array type not found`)
    const body = match[1] ?? ''
    return [...body.matchAll(/([A-Z_a-z]+)\??:/g)].map((m) => m[1] ?? '').sort()
  }
  const expected = ['actual', 'forced', 'index', 'prefill_ms', 'requested', 'tokens']
  expect(pick(src('types.ts'), 'route_chunks')).toEqual(expected)
  expect(pick(generated('types.d.ts'), 'route_chunks')).toEqual(expected)
})
