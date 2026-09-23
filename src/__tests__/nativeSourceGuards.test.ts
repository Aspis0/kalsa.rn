
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

test('eval-failure log carries no token text', () => {
  // Anchored on ', n_eval' so it cannot match the unrelated
  // 'failed to eval speaker prefix' statement earlier in the file.
  const logStatement = cpp('rn-completion.cpp').match(
    /LOG_ERROR\("failed to eval, n_eval[\S\s]*?\);/,
  )
  expect(logStatement).not.toBeNull()
  expect(logStatement![0]).not.toContain('tokens_to_str')
})
