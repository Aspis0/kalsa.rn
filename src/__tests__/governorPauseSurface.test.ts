/**
 * The pause contract as CONTROL FLOW, not fragments. Luna's sabotage made
 * the classifier `false && result == -2 …` while every `toContain` substring
 * survived, so these guards slice the native blocks between markers and
 * compare them token-for-token (comments stripped): the gate, every branch,
 * every label, the rewind reset, and the emission guard all have to match,
 * and a dead gate cannot keep its old body text under the same start marker.
 * The TS union is pinned against the GENERATED declarations, so a stale
 * `lib/typescript/types.d.ts` fails too.
 *
 * What stays unpinned without a native build: that the engine really stamps
 * rule 0 on a Wait and {2,3,9} on an admit (llama-governor-policy/runtime
 * behaviour — read, not executed), that a -2 actually flows decode →
 * completion → JSON → JS, that row+1 makes the count<=1 guard unreachable at
 * runtime, and sensor freshness. `tests/` has the native harness for those,
 * but running it is a native build.
 */
import fs from 'fs'
import path from 'path'

const cpp = (name: string) => fs.readFileSync(path.join(__dirname, '../../cpp', name), 'utf8')
const src = (name: string) => fs.readFileSync(path.join(__dirname, '../../src', name), 'utf8')

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

/** The pause_reason union of a types file, sorted. */
function pauseUnion(text: string): string[] {
  const match = text.match(/pause_reason\?:\s*([^\n;]+)/)
  if (match === null) throw new Error('pause_reason union not found')
  const union = match[1] ?? ''
  return [...union.matchAll(/'([^']+)'/g)].map((m) => m[1] ?? '').sort()
}

test('the classifier gates on the -2 and labels only from the governor\'s own stats', () => {
  expect(block(cpp('rn-llama.cpp'), 'if (result == -2 && !governor->failed())', 'return result;'))
    .toBe(
      'if (result == -2 && !governor->failed()) { ' +
        'const llama_governor_stats stats = governor->stats(); ' +
        'if (batch.n_tokens > 1) { ' +
        'if (stats.thermal_state == llama_governor_thermal_state::Unknown) { ' +
        'governor_pause_ = "profile"; ' +
        '} else if (stats.last_router_rule == 0) { ' +
        'governor_pause_ = "thermal"; ' +
        '} else { ' +
        'governor_pause_ = "unexplained"; ' +
        '} ' +
        '} else { ' +
        'governor_pause_ = stats.decode_requires_reload ? "reload" : "profile"; ' +
        '} } return result;',
    )
})

test('the getter is the literal the completion site reads', () => {
  expect(stripComments(cpp('rn-llama.h')).replace(/\s+/g, ' ')).toContain(
    'const char * governorPause() const { return governor_pause_; }',
  )
})

test('the completion records a classified pause instead of logging an eval failure — and only then', () => {
  expect(block(cpp('rn-completion.cpp'), 'const char * pause =', 'embd.resize(n_past);')).toBe(
    'const char * pause = ' +
      'decode_rc == -2 ? parent_ctx->governorPause() : nullptr; ' +
      'if (pause != nullptr) { ' +
      'governor_pause = pause; ' +
      'LOG_WARNING("governor paused decode (%s), n_eval: %d, n_past: %d", pause, n_eval, n_past); ' +
      '} else { ' +
      'LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d", ' +
      'n_eval, n_past, parent_ctx->params.cpuparams.n_threads ); ' +
      '} embd.resize(n_past);',
  )
})

test('rewind starts a completion clean: no later completion inherits a pause', () => {
  const rewind = block(
    cpp('rn-completion.cpp'),
    'void llama_rn_context_completion::rewind() {',
    'bool llama_rn_context_completion::initSampling()',
  )
  expect(rewind).toContain('incomplete = false; governor_pause = ""; n_remain = 0;')
})

test('the resolved JSON carries pause_reason only behind the empty check', () => {
  expect(
    block(cpp('jsi/JSICompletion.h'), 'if (!c.governor_pause.empty())', 'res["pause_reason"] = c.governor_pause;'),
  ).toBe(
    'if (!c.governor_pause.empty()) { res["pause_reason"] = c.governor_pause;',
  )
})

test('the TS union and the generated declarations agree — a stale d.ts fails here', () => {
  const sourceUnion = pauseUnion(src('types.ts'))
  const declaredUnion = pauseUnion(
    fs.readFileSync(path.join(__dirname, '../../lib/typescript/types.d.ts'), 'utf8'),
  )
  expect(declaredUnion).toEqual(sourceUnion)
  expect(sourceUnion).toEqual(['profile', 'reload', 'thermal', 'unexplained'])
})
