#include "rn-governor.h"
#include "llama-governor.h"
#include "rn-llama.h"

#include "ggml.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rnllama {

namespace {

// Ages of the freshness stamps in ms; -1 when the stamp is 0 (never read).
int32_t age_ms(int64_t now_ms, int64_t stamp_us) {
    return stamp_us == 0 ? -1 : static_cast<int32_t>(now_ms - stamp_us / 1000);
}

// One margin as strict JSON. A leg the reader could not see is null, never 0
// and never "nan": the campaign parser JSON.parses these payloads verbatim and
// drops one it cannot parse, and a 0 would read as a leg already at its trip.
std::string json_number(float value) {
    return std::isfinite(value) ? std::to_string(value) : "null";
}

// The leg that ran the last step, in the wire's coding: 0 cpu, 1 gpu, 2 npu,
// and 3 as the explicit gpu-coolmode marker - the engine books those decode
// tokens under decode_tokens_cpu (kalsallama llama-governor.cpp), so a 3 tells
// the replay to exclude the window from Stage 0 learning instead of silently
// mixing a coolmode GPU step into the CPU or GPU fits.
int8_t observed_leg(llama_governor_engine engine) {
    switch (engine) {
        case llama_governor_engine::CPU:
            return 0;
        case llama_governor_engine::GPU:
            return 1;
        case llama_governor_engine::NPU:
            return 2;
        case llama_governor_engine::GPU_COOLMODE:
            return 3;
    }
    return -1;
}

} // namespace

rn_governor::rn_governor(llama_model * prefill_model, llama_model * decode_model,
                         llama_context_params prefill_params, llama_context_params decode_params,
                         const llama_governor_params & params)
    : gpu_fit_(params.gpu_fit), gpu_leg_(params.gpu_leg) {
    // This is the same two-model call order as governor-bench. The v0.3 API
    // overload additionally enables the parsed policy inputs and thermo gate.
    std::string failure_reason;
    governor_ = llama_governor_init_with_params_internal(
        prefill_model, decode_model, prefill_params, decode_params, params, &failure_reason);
    if (governor_ == nullptr) {
        failure_reason_ = failure_reason.empty()
            ? "llama_governor_init_with_params returned null"
            : failure_reason;
        throw std::runtime_error(failure_reason_);
    }
    log_platform_thermal_availability();
    log_battery_sysfs();
    // The two-model form has no GPU leg, so there is no second reader to build.
    init_hop_reader(params, false);
}

rn_governor::rn_governor(llama_model * model, const llama_governor_leg * legs, uint32_t n_legs,
                         const llama_governor_params & params)
    : gpu_fit_(params.gpu_fit), gpu_leg_(params.gpu_leg) {
    // One shared llama_model, one context per leg (F1). A null engine
    // governor logged its reason on the engine side (no out-param in the
    // one-model form); the caller falls back to the two-model load.
    governor_ = llama_governor_init_one_model_with_params(model, legs, n_legs, params);
    if (governor_ == nullptr) {
        failure_reason_ = "llama_governor_init_one_model_with_params returned null";
        throw std::runtime_error(failure_reason_);
    }
    log_platform_thermal_availability();
    // A present leg the engine did not build (its device refused, e.g. the
    // OpenCL import) is a runtime demotion: logged once here, the hop simply
    // never uses that leg, and the load stays one-copy on the legs that did
    // build. The CPU leg cannot be absent - the engine refuses without it.
    // The same walk answers whether the prefill thermal sample can run at all:
    // it needs both an NPU and a GPU leg to price, and a demoted leg is one
    // the prefill handoff could never reach.
    bool npu_leg_built = false;
    bool gpu_leg_built = false;
    for (uint32_t i = 0; i < n_legs; ++i) {
        if (legs[i].engine == llama_governor_engine::CPU) {
            continue;
        }
        if (llama_governor_leg_ctx(governor_, legs[i].engine) == nullptr) {
            LOG_ERROR(
                "KALSA_GOVERNOR_FALLBACK {stage:\"onecopy_leg_absent\", models_loaded:1, engine:\"%s\"}",
                legs[i].engine == llama_governor_engine::NPU ? "NPU" : "GPU");
            continue;
        }
        npu_leg_built |= legs[i].engine == llama_governor_engine::NPU;
        gpu_leg_built |= legs[i].engine == llama_governor_engine::GPU;
    }
    log_battery_sysfs();
    init_hop_reader(params, npu_leg_built && gpu_leg_built);
}

void rn_governor::init_hop_reader(const llama_governor_params & params, bool prefill_thermal) {
    // Two readers want the zones and they start at different times: the decode
    // hop wants them when decode_hop_tokens > 0, and the prefill placement
    // wants them on every one-copy load that has both an NPU and a GPU leg to
    // price. With the hop off - the production default of 0 - the second one
    // is the only reason a phone reads a zone at all. The hop's cadence is not
    // touched by that: decode_hop_tokens_ still gates every decode-path
    // sample, and the one-time counts say which legs the headroom rules can
    // decide (both zero -> the engine's alternation fallback, e.g.
    // Jelly/MTK denies the zones).
    if (params.decode_hop_tokens > 0) {
        decode_hop_tokens_ = params.decode_hop_tokens;
    } else if (!prefill_thermal) {
        return;
    }
    decode_legs_ = std::make_unique<rn_thermal_legs>("/sys/class/thermal");
    LOG_INFO("governor: thermal legs cpu=%d npu=%d gpu=%d zones, hop_tokens=%u",
             decode_legs_->cpu_zone_count(), decode_legs_->npu_zone_count(),
             decode_legs_->gpu_zone_count(), decode_hop_tokens_);
    const rn_thermal_legs::trip_range cpu = decode_legs_->trip_range_c(rn_thermal_legs::leg::cpu);
    const rn_thermal_legs::trip_range npu = decode_legs_->trip_range_c(rn_thermal_legs::leg::npu);
    const rn_thermal_legs::trip_range gpu = decode_legs_->trip_range_c(rn_thermal_legs::leg::gpu);
    samples_.set_trips({cpu.min_c, cpu.max_c, npu.min_c, npu.max_c, gpu.min_c, gpu.max_c});
}

void rn_governor::log_battery_sysfs() const {
    const rn_battery_sysfs probe = rn_battery_sysfs_read(k_battery_sysfs_root);
    // Strict JSON: the campaign parser JSON.parses the payload verbatim
    // (telemetryParse.mjs) and silently drops what does not parse.
    LOG_INFO("KALSA_GOVERNOR_BATTERY_SYSFS "
             "{\"temp_ok\":%d, \"temp_tenths_c\":%d, \"status_ok\":%d, \"status\":%d}",
             probe.temp_ok ? 1 : 0, probe.temp_tenths_c, probe.status_ok ? 1 : 0,
             probe.status);
}

void rn_governor::log_platform_thermal_availability() const {
#if defined(__ANDROID__)
    if (const char * reason = platform_thermal_.unavailable_reason()) {
        LOG_INFO("KALSA_GOVERNOR_PLATFORM {source:\"unavailable\", reason:\"%s\"}", reason);
    }
    // Its own prefix, so KALSA_GOVERNOR_PLATFORM stays a status-only line.
    if (const char * reason = platform_thermal_.headroom_unavailable_reason()) {
        LOG_INFO("KALSA_GOVERNOR_HEADROOM {available:0, reason:\"%s\"}", reason);
    } else {
        LOG_INFO("KALSA_GOVERNOR_HEADROOM {available:1}");
    }
#endif
}

rn_governor::~rn_governor() {
    llama_governor_free(governor_);
    governor_ = nullptr;
}

llama_context * rn_governor::active_ctx() const {
    if (governor_ == nullptr) {
        return nullptr;
    }
    auto * active = llama_governor_get_context(governor_);
    return active != nullptr ? active : llama_governor_get_prefill_context(governor_);
}

llama_context * rn_governor::prefill_ctx() const {
    return governor_ == nullptr ? nullptr : llama_governor_get_prefill_context(governor_);
}

int32_t rn_governor::decode(llama_batch batch) {
    if (failed_ || governor_ == nullptr) {
        failed_ = true;
        failure_reason_ = "governor is failed";
        return -1;
    }

    // Ahead of this step's admission, because the engine's platform floor
    // (apply_platform_floor) is read at admission time and decode is the one
    // place every step of every governor path passes through.
    refresh_platform_thermal();

    int32_t result = 0;
    {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        result = llama_governor_decode(governor_, batch);
        // Observation commits only after the step ran: rc 0 is the one
        // outcome where the single-token step actually ran (-2 is flow
        // control: no step, and a paused first batch must leave the record
        // empty). The stamp after the call bounds the steps' compute time.
        if (decode_legs_ != nullptr && batch.n_tokens == 1 && result == 0) {
            commit_window(ggml_time_us() / 1000);
        }
    }
    if (governor_decode_failed(result, governor_->is_failed())) {
        failed_ = true;
        const char * reason = governor_->failure_reason();
        if (reason != nullptr) {
            failure_reason_ = reason;
        } else {
            failure_reason_ = "llama_governor_decode failed with rc=" + std::to_string(result);
        }
    }
    return result;
}

bool rn_governor::trim_sequence(llama_pos p) const {
    if (governor_ == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return llama_governor_trim_sequence(governor_, p);
}

bool rn_governor::engine_failed() const {
    return governor_ != nullptr && governor_->is_failed();
}

void rn_governor::clear_cache(bool clear_data) {
    // The engine restarts its hop clock here (llama-governor-runtime.cpp
    // clear_cache); keep the mirror in step. Both happen under engine_mutex_,
    // which before_decode_batch also holds for the mirror it advances. The
    // observation record restarts with it: the mirror's next 0 opens a first
    // window again, and a ring from before the clear would answer for a hop
    // clock that no longer exists.
    std::lock_guard<std::mutex> lock(engine_mutex_);
    decode_tokens_since_prefill_ = 0;
    reset_observation_record_locked();
    // The engine drops its prefill margins here (clear_cache calls
    // clear_prefill_headroom), so the next turn must sample again.
    prefill_headroom_sampled_ = false;
    llama_governor_clear_cache(governor_, clear_data);
}

void rn_governor::reset_prefill_stats() {
    // The engine restarts its hop clock here too (the completion's
    // beginCompletion is one caller), and it drops the prefill margins with it
    // (reset_prefill_stats calls clear_prefill_headroom); keep both mirrors in
    // step, so a turn that never prefills cannot leave last turn's sample.
    std::lock_guard<std::mutex> lock(engine_mutex_);
    decode_tokens_since_prefill_ = 0;
    prefill_headroom_sampled_ = false;
    // The record is per completion: this is its start.
    reset_observation_record_locked();
    llama_governor_reset_prefill_stats(governor_);
}

void rn_governor::before_decode_batch(int n_tokens) {
    if (decode_legs_ == nullptr) {
        return;
    }
    // The mirror is read and written here and reset by clear_cache from a JSI
    // worker, so the whole step, the leg sample included, runs under the lock.
    std::lock_guard<std::mutex> lock(engine_mutex_);
    // The engine's decode_tokens_since_prefill_ counts exactly these
    // standard-path batches (multi-token batches are prefills that restart
    // it; 1-token batches increment it after the run), and select_decode
    // decides window k on the pre-increment count k*N - so feeding when the
    // mirror hits a multiple of N puts a fresh sample under every window
    // decision, first window included.
    if (n_tokens > 1) {
        decode_tokens_since_prefill_ = 0;
        // The prefill restarts the hop clock; a boundary staged for the
        // abandoned sequence must never commit.
        pending_valid_ = false;
        // The prefill placement prices the legs off this one sample, so it
        // runs on the turn's first prefill batch and not on every chunk.
        if (!prefill_headroom_sampled_) {
            prefill_headroom_sampled_ = true;
            push_prefill_headroom();
        }
        return;
    }
    if (decode_hop_tokens_ == 0) {
        // The reader exists for the prefill sample alone (see
        // init_hop_reader); the hop has no window to stage.
        return;
    }
    const int64_t now_ms = ggml_time_us() / 1000;
    if (decode_tokens_since_prefill_ % decode_hop_tokens_ == 0) {
        const rn_thermal_legs::headroom legs = feed_decode_headroom();
        stage_window(now_ms, legs);
    }
    ++decode_tokens_since_prefill_;
}

// The NPU and GPU margins, in C to each leg's first passive trip, pushed to
// the engine's prefill placement (llama_governor_set_prefill_headroom) once
// per turn. A leg with no usable zone reads NaN there, which the engine takes
// as unknown and answers by keeping prefill on the NPU. Caller holds
// engine_mutex_.
void rn_governor::push_prefill_headroom() {
    const rn_thermal_legs::headroom legs = decode_legs_->sample();
    llama_governor_set_prefill_headroom(governor_, legs.npu_headroom_c, legs.gpu_headroom_c);
    // Pushed whether or not prefill_thermal_handoff is on: with the switch
    // false the engine's rule stays off, and the line is what the device
    // check reads to prove the reader is built and the margins are finite.
    const std::string npu_json = json_number(legs.npu_headroom_c);
    const std::string gpu_json = json_number(legs.gpu_headroom_c);
    LOG_INFO("KALSA_GOVERNOR_PREFILL_THERMAL {\"npu_headroom_c\":%s, \"gpu_headroom_c\":%s}",
             npu_json.c_str(), gpu_json.c_str());
}

void rn_governor::stage_window(int64_t now_ms, const rn_thermal_legs::headroom & legs) {
    rn_window_sample sample;
    sample.tokens = decode_tokens_since_prefill_;
    sample.headroom = observed_headroom_;
    sample.headroom_age_ms = age_ms(now_ms, observed_headroom_read_us_);
    sample.status = observed_status_.load(std::memory_order_relaxed);
    sample.status_age_ms = age_ms(now_ms, observed_status_read_us_.load(std::memory_order_relaxed));
    sample.app_age_ms = age_ms(now_ms, observed_app_sample_us_.load(std::memory_order_relaxed));
    sample.cpu_hot_c = legs.cpu_hot_c;
    sample.npu_hot_c = legs.npu_hot_c;
    sample.gpu_hot_c = legs.gpu_hot_c;
    if (decode_tokens_since_prefill_ == 0) {
        // The completion's first window: no step of this completion has run,
        // so the engine's cumulative leg/duty/applied values belong to the
        // previous completion and stay unknown. One battery sysfs probe per
        // completion, staged here and committed with the first step.
        pending_battery_ = rn_battery_sysfs_read(k_battery_sysfs_root);
        pending_first_window_ = true;
    } else {
        // The engine's "last" values are exactly the inputs that produced
        // duty_last: pairing them here pairs duty with what it realized on,
        // not with this boundary's snapshot.
        llama_governor_stats stats{};
        llama_governor_get_stats(governor_, &stats);
        sample.leg_prev = observed_leg(stats.decode_engine);
        sample.applied_headroom = stats.decode_pacing_headroom_last;
        sample.applied_status = stats.platform_thermal_status;
        sample.duty_last = stats.decode_pacing_duty_last;
    }
    pending_sample_ = sample;
    pending_boundary_ms_ = now_ms;
    pending_valid_ = true;
}

void rn_governor::commit_window(int64_t step_end_ms) {
    // Only a step that ran (the rc 0 gate at the caller) may touch the
    // record: a paused first batch leaves the stamps 0 and the ring empty
    // (types.ts: "0 when no step has run"), and the ring length tracks
    // tokens that actually ran.
    samples_.last_step(step_end_ms);
    if (!pending_valid_) {
        return;
    }
    if (pending_first_window_) {
        samples_.first_step(pending_boundary_ms_, pending_sample_, pending_battery_);
        pending_first_window_ = false;
    }
    samples_.record(pending_sample_);
    pending_valid_ = false;
}

void rn_governor::reset_observation_record() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    reset_observation_record_locked();
}

void rn_governor::reset_observation_record_locked() {
    samples_.reset();
    pending_valid_ = false;
    pending_first_window_ = false;
}

rn_thermal_legs::headroom rn_governor::feed_decode_headroom() {
    if (decode_legs_ == nullptr) {
        return {};
    }
    const rn_thermal_legs::headroom legs = decode_legs_->sample();
    // The GPU value is NaN without gpu zones (no GPU leg): the engine's hop
    // treats an unknown leg as unavailable, which is exactly the two-model
    // and Jelly behaviour.
    llama_governor_set_decode_headroom(governor_, legs.cpu_headroom_c, legs.npu_headroom_c,
                                       legs.gpu_headroom_c);
    return legs;
}

bool rn_governor::set_thermo_profile(const llama_governor_thermo_profile & profile) {
    if (governor_ == nullptr || failed_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(profile_mutex_);
    // -1 is "no app vote" (key absent, malformed, or the app's read failed).
    // That is a gap, not a cool-down, so the remembered status survives it
    // while a native read returned a status in the last 10 s. Native reads
    // run only inside a completion, so a profile sent after a longer pause
    // drops the carry and the native reader re-escalates (5 s debounce);
    // the engine's COOLMODE dwell keeps an entered COOLMODE meanwhile.
    llama_governor_thermo_profile merged = profile;
    const bool native_recent = platform_native_valid_us_ > 0 &&
        ggml_time_us() - platform_native_valid_us_ <= k_platform_thermal_carry_us;
    if (merged.platform_thermal_status < 0 && native_recent) {
        merged.platform_thermal_status = thermo_profile_.platform_thermal_status;
    }
    // The engine reads a negative headroom as unknown (normalize_thermal_headroom),
    // so an app profile without one takes the native read the same way.
    if (!(merged.platform_thermal_headroom >= 0.0f)) {
        merged.platform_thermal_headroom =
            rn_platform_thermal_headroom_carried(ggml_time_us(), platform_headroom_state_);
    }
    const bool status_changed =
        merged.platform_thermal_status != thermo_profile_.platform_thermal_status;
    {
        std::lock_guard<std::mutex> engine_lock(engine_mutex_);
        profile_valid_ = llama_governor_set_thermo_profile(governor_, merged, ggml_time_us() / 1000);
    }
    if (profile_valid_) {
        thermo_profile_ = merged;
        // The app profile carries the battery inputs: this is the app sample
        // the window observation ages.
        observed_app_sample_us_.store(ggml_time_us(), std::memory_order_relaxed);
        // An app profile cancels a pending native escalation only when it
        // changes the status.
        if (status_changed) {
            platform_thermal_send_state_ = {};
        }
        platform_refusal_logged_ = false;
    }
    return profile_valid_;
}

void rn_governor::refresh_platform_thermal() {
    bool status_due = false;
    bool headroom_due = false;
    {
        std::lock_guard<std::mutex> lock(profile_mutex_);
        if (!profile_valid_) {
            return;
        }
        const int64_t now_us = ggml_time_us();
        status_due = rn_platform_thermal_should_read(now_us, platform_thermal_read_us_,
                                                     k_platform_thermal_interval_us);
        headroom_due = rn_platform_thermal_should_read(now_us, platform_headroom_read_us_,
                                                       k_platform_thermal_headroom_interval_us);
    }
    if (!status_due && !headroom_due) {
        return;
    }
    // The HAL calls can block on binder; profile updates remain available while
    // they run, and the send decisions use whichever profile was accepted latest.
    // A status that is not due reads as absent, which the send decision ignores.
    const int32_t status = status_due ? platform_thermal_.status() : k_platform_thermal_absent;
    const float headroom_read = headroom_due ? platform_thermal_.headroom()
                                             : std::numeric_limits<float>::quiet_NaN();
    std::lock_guard<std::mutex> lock(profile_mutex_);
    if (!profile_valid_) {
        return;
    }
    if (status != k_platform_thermal_absent) {
        const int64_t now_us = ggml_time_us();
        platform_native_valid_us_ = now_us;
        observed_status_.store(status, std::memory_order_relaxed);
        observed_status_read_us_.store(now_us, std::memory_order_relaxed);
    }
    const int32_t previous = thermo_profile_.platform_thermal_status;
    const auto decision = rn_platform_thermal_should_send(
        ggml_time_us(), status, previous, platform_thermal_send_state_);
    platform_thermal_send_state_ = decision.next_state;
    if (decision.pending_started) {
        LOG_INFO("KALSA_GOVERNOR_PLATFORM {status:%d, prev:%d, source:\"native\", sent:0}",
                 status, previous);
    }
    const float previous_headroom = thermo_profile_.platform_thermal_headroom;
    float headroom = previous_headroom;
    if (headroom_due) {
        headroom = rn_platform_thermal_headroom_forward(ggml_time_us(), headroom_read,
                                                        platform_headroom_state_);
        // Freshness is a property of the read, not of a change: a stable
        // forecast is re-read every 2 s, and the early return below must not
        // make it look ever staler on the observation wire. The value follows
        // what the engine would see (NaN once the carry expires); the stamp
        // advances only on a finite read, which is exactly when
        // rn_platform_thermal_headroom_forward refreshes read_us. Both go
        // under engine_mutex_ - the lock the window staging reads them under -
        // taken here under profile_mutex_, the sanctioned order, so a sample
        // can never pair one read's value with another read's stamp.
        {
            std::lock_guard<std::mutex> engine_lock(engine_mutex_);
            observed_headroom_ = headroom;
            if (std::isfinite(headroom)) {
                observed_headroom_read_us_ = platform_headroom_state_.read_us;
            }
        }
    }
    // NaN is a value here too: two unknowns are the same headroom.
    const bool headroom_changed = std::isnan(headroom) ? !std::isnan(previous_headroom)
                                                       : headroom != previous_headroom;
    if (!decision.send && !headroom_changed) {
        return;
    }
    llama_governor_thermo_profile updated = thermo_profile_;
    if (decision.send) {
        updated.platform_thermal_status = status;
    }
    updated.platform_thermal_headroom = headroom;
    bool accepted = false;
    {
        std::lock_guard<std::mutex> engine_lock(engine_mutex_);
        accepted = llama_governor_set_thermo_profile(governor_, updated, ggml_time_us() / 1000);
    }
    if (!accepted) {
        profile_valid_ = false;
        platform_thermal_send_state_ = {};
        if (!platform_refusal_logged_) {
            LOG_WARNING("KALSA_GOVERNOR_PLATFORM {source:\"native\", sent:0, reason:\"engine_refused\"}");
            platform_refusal_logged_ = true;
        }
        return;
    }
    thermo_profile_ = updated;
    if (decision.send) {
        LOG_INFO("KALSA_GOVERNOR_PLATFORM {status:%d, prev:%d, source:\"native\", sent:1}",
                 status, previous);
    }
}

bool rn_governor::profile_valid() const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    return profile_valid_;
}

bool rn_governor::set_prefill_override(int mode) {
    if (governor_ == nullptr || failed_) {
        return false;
    }
    return llama_governor_set_prefill_override(governor_, mode);
}

llama_governor_stats rn_governor::stats() const {
    llama_governor_stats result{};
    std::lock_guard<std::mutex> lock(engine_mutex_);
    llama_governor_get_stats(governor_, &result);
    return result;
}

std::pair<llama_governor_stats, rn_governor_samples> rn_governor::stats_with_samples() const {
    // One engine_mutex_ hold for both copies: taking stats() and samples()
    // separately lets a decode step land between them, pairing one
    // completion's stats with the next one's record.
    std::lock_guard<std::mutex> lock(engine_mutex_);
    llama_governor_stats stats{};
    llama_governor_get_stats(governor_, &stats);
    return {stats, samples_};
}

} // namespace rnllama
