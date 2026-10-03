/**
 * Source-text guards for the runtime HTP fallback (jest cannot execute the
 * native decode path): the recorder must ride the one real-failure branch of
 * the decode wrapper, degrade only an engaged lane, and keep the published
 * plan reason out of environ — the app's reload then lands off HTP through
 * the same KALSA_HTP_FALLBACK machinery an init failure uses.
 */
import fs from 'fs'
import path from 'path'

const rnLlama = fs.readFileSync(path.join(__dirname, '../../cpp/rn-llama.cpp'), 'utf8')
const rnLlamaHeader = fs.readFileSync(path.join(__dirname, '../../cpp/rn-llama.h'), 'utf8')
const rnLlamaJsi = fs.readFileSync(path.join(__dirname, '../../cpp/jsi/RNLlamaJSI.cpp'), 'utf8')

const recorder = () =>
  rnLlama.match(/void llama_rn_context::note_htp_runtime_fallback\(\) {[\S\s]*?\n}/)

test('a real governor decode failure records the runtime fallback', () => {
  // Inside the only real-failure branch (rc fails the engine-state
  // discriminator and the failure is fresh), before the KALSA_GOVERNOR_FALLBACK
  // log the app's scan keys on — never on the flow-control -2 path. The
  // recorder rides the HTP attribution gate (see the pure helper's contract):
  // a KV-commit or CPU-routed failure must not kill the lane.
  expect(rnLlama).toMatch(
    /governor_decode_failed\(result, governor->engine_failed\(\)\) && !was_failed\) {\s*\n\s*const std::string & reason = governor->failure_reason\(\);[\S\s]*?htp_runtime_failure\([\S\s]*?reason\.c_str\(\)\)\) {\s*\n\s*note_htp_runtime_fallback\(\);/,
  )
})

test('the recorder degrades only an engaged, not-yet-degraded lane', () => {
  const fn = recorder()
  expect(fn).not.toBeNull()
  // governorNpuDevice() is non-empty only when the lane resolved to HTP0; a
  // recorded reason means an earlier degrade (init or runtime) already won.
  expect(fn![0]).toContain('governorNpuDevice().empty()')
  // First writer wins under the mutex: the stats task reads the reason from
  // a JSI worker thread while a decode may be recording it.
  expect(fn![0]).toContain('!governor_npu_fallback_.empty()')
  expect(fn![0]).toContain('std::lock_guard<std::mutex>')
})

test('the recorder sets the env with overwrite, never a parallel route path', () => {
  const fn = recorder()
  expect(fn).not.toBeNull()
  // Overwrite: a stale value from an earlier attempt must not win.
  expect(fn![0]).toContain('setenv("KALSA_HTP_FALLBACK", KALSA_HTP_RUNTIME_FALLBACK')
  expect(fn![0]).toContain('/*overwrite=*/1')
  // The governor is sticky-failed here; flipping the route override on this
  // context pair would be a dead write and a parallel path beside the env.
  expect(fn![0]).not.toContain('set_prefill_override')
})

test('each load reads the env fresh; the plan fields are mutex-guarded', () => {
  // No caching of the reason across loads: the runtime setenv must be seen
  // by the next load_governor_models.
  expect(rnLlama).toContain('std::getenv("KALSA_HTP_FALLBACK")')
  // setenv(3) may reallocate environ, so nothing holds a getenv pointer: the
  // load copies the reason bytes through the synchronized accessor, and the
  // stats task reads a copy through the same mutex.
  expect(rnLlama).toContain('owner.setGovernorNpuDevice(device_plan.npu_device);')
  expect(rnLlama).toContain('owner.setGovernorNpuFallback(device_plan.npu_fallback);')
  expect(rnLlamaHeader).toContain('std::mutex npu_fallback_mutex_')
  expect(rnLlamaHeader).toContain('std::string governorNpuDevice() const;')
  expect(rnLlamaHeader).toContain('std::string governorNpuFallback() const;')
})

test('the stats task reads the device through the same guarded accessor', () => {
  // The bare ctx->governor_npu_device read was the sibling of the fallback
  // race: bytes copied through an unsynchronized pointer on a JSI worker.
  expect(rnLlamaJsi).toContain('ctx->governorNpuDevice()')
  expect(rnLlamaJsi).not.toContain('ctx->governor_npu_device')
})

test('the recorded reason rides the log line the lab parses, at failure time', () => {
  // The end-of-turn KALSA_GOVERNOR telemetry is skipped on the throwing turn
  // and the retry's reload suppresses the plan line, so the
  // KALSA_GOVERNOR_FALLBACK line is the only surface that fires when the
  // failure happens; it must carry the recorded constant, not only the
  // engine's raw reason.
  expect(rnLlama).toMatch(/KALSA_GOVERNOR_FALLBACK {[^]*?npu_fallback:\\"%s\\"/)
  expect(rnLlama).toMatch(/npu_fallback:\\"%s\\", gpu_fit:%d/)
})
