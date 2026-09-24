/**
 * Source guards for the governor thermal-pause outcome: jest cannot execute
 * the native completion, so the contract is pinned where it is written —
 * classification in the decode funnel, the completion field, the JSON surface,
 * and the guarantee that a pause never becomes a governor failure (the
 * "Governor decode failed: " throw stays reserved for real failures and is
 * guarded separately in nativeSourceGuards.test.ts).
 */
import fs from 'fs'
import path from 'path'

const cpp = (name: string) => fs.readFileSync(path.join(__dirname, '../../cpp', name), 'utf8')

test('decode classifies a flow-control -2 only while the governor did not fail', () => {
  const source = cpp('rn-llama.cpp')
  // Reset at entry so a reader after its own decode sees that call's outcome.
  expect(source).toMatch(/governor_pause_ = nullptr;\n\s*if \(!governor\)/)
  expect(source).toContain('result == -2 && !governor->failed()')
  // The three typed values, from the batch kind + the stats refreshed at
  // this decode's entry.
  expect(source).toContain('"thermal"')
  expect(source).toContain('"profile"')
  expect(source).toContain('"reload"')
  expect(source).toContain('stats.decode_requires_reload')
  expect(source).toContain('llama_governor_thermal_state::Unknown')
})

test('completion records the pause instead of logging an eval failure', () => {
  const source = cpp('rn-completion.cpp')
  expect(source).toContain('const char * pause =')
  expect(source).toContain('decode_rc == -2 ? parent_ctx->governorPause() : nullptr')
  expect(source).toContain('governor_pause = pause;')
  // The real-failure statement must survive for non-pause rc (its own guard
  // in nativeSourceGuards.test.ts anchors on it).
  expect(source).toMatch(/LOG_ERROR\("failed to eval, n_eval/)
})

test('the resolved result carries pause_reason only when paused', () => {
  const source = cpp('jsi/JSICompletion.h')
  expect(source).toContain('if (!c.governor_pause.empty())')
  expect(source).toContain('res["pause_reason"] = c.governor_pause;')
  // The throw channel for real failures is untouched and untouched-for-pauses:
  // a pause never latches the failed state, so this gate cannot see one.
  const jsi = cpp('jsi/RNLlamaJSI.cpp')
  expect(jsi).toContain('"Governor decode failed: "')
  expect(jsi).toContain('if (ctx->governorFailed())')
})

test('the TS surface types the three pause reasons', () => {
  const types = fs.readFileSync(path.join(__dirname, '../../src/types.ts'), 'utf8')
  expect(types).toContain(
    "pause_reason?: 'thermal' | 'profile' | 'reload'",
  )
})
