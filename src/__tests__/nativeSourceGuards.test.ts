/**
 * Source-text guards for binding-native invariants jest cannot execute: the
 * governor failure error prefix is a stable API (the JSI throw is the delivery
 * channel to JS), and the eval-failure log must not carry prompt text into
 * logcat.
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
