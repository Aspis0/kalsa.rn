#include "dense_weights.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace bmoe {

using clock_t_ = std::chrono::steady_clock;

// OpenCL (and any other device) buffers are not host-visible. Rebinding tensor->data off them
// undoes -ngl and allocates a second DRAM copy of the same bytes.
static bool tensor_is_host(const ggml_tensor * t) {
    return !t || !t->buffer || ggml_backend_buffer_is_host(t->buffer);
}

// Routed expert weights: blk.N.ffn_{up,down,gate,gate_up}_{,ch}exps — not *_shexp, not the
// router. The streamer owns these; AnonGpu must never rebind or DONTNEED them.
static bool is_routed_expert_name(const char * name) {
    if (!name || !name[0]) return false;
    const std::string n(name);
    if (n.find("shexp") != std::string::npos) return false;
    if (n.find(".ffn_") == std::string::npos) return false;
    return n.find("exps") != std::string::npos;
}

DenseWeights::~DenseWeights() {
    shutdown();
}

std::vector<std::pair<uint64_t, uint64_t>>
DenseWeights::byte_ranges(std::vector<std::pair<uint64_t, uint64_t>> expert_ranges, uint64_t file_size) {
    std::sort(expert_ranges.begin(), expert_ranges.end());
    std::vector<std::pair<uint64_t, uint64_t>> dense;
    uint64_t pos = 0;
    for (const auto & r : expert_ranges) {
        if (r.first > pos) dense.push_back({pos, r.first}); // the gap before this expert range is dense
        pos = std::max(pos, r.second);
    }
    if (pos < file_size) dense.push_back({pos, file_size}); // the trailing dense tail (lm_head et al.)
    return dense;
}

bool DenseWeights::init(DenseWeightsMode mode,
                        const std::vector<std::string> & paths,
                        size_t align,
                        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges,
                        std::vector<DenseTensorRef> tensors) {
    mode_ = mode;
    paths_ = paths;
    align_ = align ? align : 4096;
    ranges_ = std::move(ranges);
    tensors_ = std::move(tensors);
    basenames_.clear();
    for (const std::string & p : paths_) {
        const size_t slash = p.find_last_of("/\\");
        basenames_.push_back(slash == std::string::npos ? p : p.substr(slash + 1));
    }

    if (dense_mode_host_copy(mode_)) {
        if (tensors_.empty()) return true; // nothing captured to rebind — behave as Mmap
        if (mode_ == DenseWeightsMode::Pinned && pio::pinned_max_bytes() == 0) {
            std::fprintf(stderr, "bmoe: --dense-weights ahwb needs reclaim-exempt memory, which this "
                                 "platform does not provide (Android only)\n");
            return false;
        }
        uint64_t gpu_skip = 0;
        bool any_host = false;
        for (const DenseTensorRef & d : tensors_) {
            if (!d.tensor || d.size == 0) continue;
            if (is_routed_expert_name(d.tensor->name)) continue;
            if (tensor_is_host(d.tensor))
                any_host = true;
            else
                gpu_skip += d.size;
        }
        if (mode_ == DenseWeightsMode::AnonGpu && gpu_skip == 0)
            std::fprintf(stderr, "bmoe: dense-weights=anon-gpu — no GPU-resident dense, behaving as anon\n");
        if (gpu_skip)
            std::fprintf(stderr, "bmoe: dense-weights=%s — skipping %llu MiB already on GPU (no host twin)\n",
                         dense_weights_flag(mode_), (unsigned long long) (gpu_skip >> 20));
        if (any_host) {
            // Single-lane readers (one per shard) with a bounce large enough for our chunk; O_DIRECT
            // independent of the expert stream. Sized to the largest tensor is unnecessary — we read
            // in bounded chunks.
            const size_t chunk = 8ull << 20;
            for (const std::string & p : paths_) {
                readers_.push_back(std::unique_ptr<FileReader>(new FileReader()));
                if (!readers_.back()->open(p, 1, /*direct=*/true, align_, chunk + 2 * align_)) return false;
            }
            if (!read_anonymous(align_)) return false;
            // The tensors are copied and rebound; nothing reads through these again. Their fds and
            // per-lane bounce buffers would otherwise sit allocated for the whole session, next to
            // the expert cache that is counting every MiB.
            readers_.clear();
        }
        // Always drop leftover file pages, including those of GPU-copied dense that sit between
        // CPU expert tensors. llama.cpp does not unmap that window after ngl>0.
        drop_mmap_copies(pio::vm_page());
    } else if (mode_ == DenseWeightsMode::Warmed) {
        warm();
    }
    return true;
}

// ── Anonymous / Pinned: read each dense tensor whole into our own buffer and rebind onto it ──
//
// The two modes differ ONLY in where the buffer comes from: ordinary anonymous memory, which the
// kernel may reclaim to zram, or a dma-buf it may not touch at all. Everything else — the fill of
// the buffer (memcpy from resident mmap pages, else O_DIRECT), the rebind, handing back the now-
// unreferenced mmap pages — is identical, which is what makes the A/B between them a single-
// variable experiment rather than two code paths being compared.
//
// Capture warm-up (and any residual page cache) often already holds the dense trunk in RAM. When
// tensor->data still points into the file mmap, probe residency PAGE BY PAGE (mincore) and:
//   - resident page → memcpy into the anon buffer (zero flash);
//   - non-resident page → O_DIRECT for that page-range only.
// No global fallback: a single reclaimed page never forces the rest of a tensor back to flash.
// Sub-page tensors probe their containing page; if resident, the whole tensor is memcpy'd.
//
// Pinned allocates PER TENSOR, so the 2047 MiB ceiling on a single dma-buf is not a constraint in
// practice: the largest dense tensor here is an embedding or lm_head, far below it. A tensor that
// did exceed it fails the run rather than quietly taking an anon buffer, because a silent mix would
// make the comparison meaningless in exactly the direction that flatters the feature.
bool DenseWeights::read_anonymous(size_t align) {
    const bool pinned = mode_ == DenseWeightsMode::Pinned;
    const size_t page = pio::vm_page();
    uint64_t total = 0;
    uint64_t memcpy_bytes = 0;
    uint64_t odirect_bytes = 0;
    uint64_t ranges_fallback = 0;
    resolve_vmas(); // mmap address of each dense tensor (for residency probe + memcpy source)
    bufs_.reserve(tensors_.size());
    buf_sz_.reserve(tensors_.size());
    if (pinned) pinned_.reserve(tensors_.size());

    // Per-tensor working state so we can two-pass: (1) memcpy every page still resident NOW,
    // before progressive anon alloc reclaims the rest; (2) O_DIRECT only the holes.
    struct Item {
        const DenseTensorRef * d = nullptr;
        void * buf = nullptr;
        const char * mmap_src = nullptr; // nullptr → no mapping, full O_DIRECT
    };
    std::vector<Item> items;
    items.reserve(tensors_.size());

    auto alloc_one = [&](const DenseTensorRef & d, void ** out) -> bool {
        if (pinned) {
            pio::PinnedAlloc pa;
            if (!pio::pinned_alloc((size_t) d.size, &pa)) {
                std::fprintf(stderr, "bmoe: pinned dense buffer %llu MiB failed (ceiling %llu MiB)\n",
                             (unsigned long long) (d.size >> 20),
                             (unsigned long long) (pio::pinned_max_bytes() >> 20));
                return false;
            }
            pinned_.push_back(pa);
            *out = pa.base;
        } else {
            void * b = pio::alloc_aligned(align, (size_t) d.size);
            if (!b) {
                std::fprintf(stderr, "bmoe: dense buffer alloc %llu failed\n", (unsigned long long) d.size);
                return false;
            }
            bufs_.push_back(b);
            *out = b;
        }
        return true;
    };

    auto mmap_src_of = [&](const DenseTensorRef & d) -> const char * {
        // Prefer tensor->data when it already points into a mapped VMA of this shard.
        if (d.tensor->data && d.file_idx >= 0 && d.file_idx < (int) vmas_.size()) {
            const uintptr_t p = (uintptr_t) d.tensor->data;
            for (const auto & v : vmas_[(size_t) d.file_idx]) {
                if (p >= v.start && p < v.end) return (const char *) d.tensor->data;
            }
        }
        return addr_of(d.file_idx, d.file_off);
    };

    // Page-granular residency of [src, src+n): true only if every fully-covered page is resident.
    // Sub-page spans probe their containing page so small norms are not forced to flash.
    auto range_resident = [&](const char * src, size_t n) -> bool {
        if (!src || n == 0) return false;
        if (n < page) {
            size_t s = 0, r = 0;
            const char * pg = (const char *) ((uintptr_t) src & ~(uintptr_t) (page - 1));
            return pio::vm_resident_sample(pg, page, &s, &r) && s > 0 && r == s;
        }
        size_t s = 0, r = 0;
        if (!pio::vm_resident_sample(src, n, &s, &r) || s == 0) return false;
        return r == s;
    };

    // ── pass 0a: resolve mmap sources WITHOUT allocating (mincore sees post-warmup state) ─
    for (const DenseTensorRef & d : tensors_) {
        if (!d.tensor || d.size == 0) continue;
        if (is_routed_expert_name(d.tensor->name)) continue; // streamer owns these
        if (!tensor_is_host(d.tensor)) continue; // already one DRAM copy (the GPU buffer)
        if (d.file_idx < 0 || d.file_idx >= (int) readers_.size()) {
            std::fprintf(stderr, "bmoe: dense tensor points at shard %d of %zu\n", d.file_idx, readers_.size());
            return false;
        }
        Item it;
        it.d = &d;
        it.mmap_src = mmap_src_of(d);
        items.push_back(it);
        total += d.size;
    }

    // ── pass 0b: plan resident runs vs holes from a single mincore sweep ─────────────
    // Recording the plan before any anon alloc means progressive buffer commits cannot
    // reclaim pages we already classified as resident.
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> copies(items.size()); // resident [off,end)
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> holes(items.size());  // non-resident
    for (size_t i = 0; i < items.size(); ++i) {
        Item & it = items[i];
        const DenseTensorRef & d = *it.d;
        if (!it.mmap_src) {
            holes[i].push_back({0, d.size});
            continue;
        }
        uint64_t done = 0;
        while (done < d.size) {
            const uint64_t remain = d.size - done;
            const size_t max_run = page * 512;
            size_t run = (size_t) std::min<uint64_t>(remain, page);
            if (remain >= page) {
                const uintptr_t a = (uintptr_t) (it.mmap_src + done);
                const uintptr_t next_pg = (a + page) & ~(uintptr_t) (page - 1);
                const uint64_t to_pg = (uint64_t) (next_pg - a);
                if (to_pg && to_pg <= remain) run = (size_t) to_pg;
            }
            const bool pol = range_resident(it.mmap_src + done, run);
            while (run < remain && run < max_run) {
                const size_t step = (size_t) std::min<uint64_t>(page, remain - run);
                if (range_resident(it.mmap_src + done + run, step) != pol) break;
                run += step;
            }
            if (pol) copies[i].push_back({done, done + run});
            else holes[i].push_back({done, done + run});
            done += run;
        }
    }

    // ── pass 1: allocate + memcpy planned resident runs + O_DIRECT holes ────────────
    for (size_t i = 0; i < items.size(); ++i) {
        Item & it = items[i];
        const DenseTensorRef & d = *it.d;
        if (!alloc_one(d, &it.buf)) return false;
        bases_.push_back(it.buf);
        buf_sz_.push_back((size_t) d.size);
        for (const auto & c : copies[i]) {
            const uint64_t off = c.first;
            const uint64_t n = c.second - c.first;
            std::memcpy((char *) it.buf + off, it.mmap_src + off, (size_t) n);
            memcpy_bytes += n;
        }
        for (const auto & h : holes[i]) {
            const uint64_t off = h.first;
            const uint64_t n = h.second - h.first;
            if (n == 0) continue;
            if (readers_[(size_t) d.file_idx]->read(0, (char *) it.buf + off, d.file_off + off, n) < 0)
                return false;
            odirect_bytes += n;
            ++ranges_fallback;
        }
        d.tensor->data = it.buf; // rebind after the buffer is fully filled
    }

    std::fprintf(stderr, "bmoe: dense-weights=%s — %llu MiB in %zu %s buffers\n", dense_weights_flag(mode_),
                 (unsigned long long) (total >> 20), buf_sz_.size(), pinned ? "pinned" : "anon");
    // Permanent instrumentation: how the anon fill was sourced (G-D2 signature of pezzo 1).
    std::fprintf(stderr, "bmoe: dense-anon: memcpy=%llu odirect=%llu ranges_fallback=%llu\n",
                 (unsigned long long) (memcpy_bytes >> 20), (unsigned long long) (odirect_bytes >> 20),
                 (unsigned long long) ranges_fallback);
    return true;
}

// Per-shard VMA resolution for every consumer that must turn (file_idx, offset) into an address.
// llama.cpp maps each shard of a split model separately, so the lookup is per shard basename.
void DenseWeights::resolve_vmas() {
    if (vmas_tried_) return;
    vmas_tried_ = true;
    vmas_.resize(basenames_.size());
    for (size_t s = 0; s < basenames_.size(); ++s)
        pio::file_mapped_regions(basenames_[s].c_str(), vmas_[s]);
}

const char * DenseWeights::addr_of(int file_idx, uint64_t off) const {
    if (file_idx < 0 || file_idx >= (int) vmas_.size()) return nullptr;
    for (const auto & v : vmas_[file_idx]) {
        const uint64_t span = (uint64_t) (v.end - v.start);
        if (off >= v.file_offset && off < v.file_offset + span) return (const char *) v.start + (off - v.file_offset);
    }
    return nullptr;
}

// Hand the mmap copies back. The capture warm-up decode faulted these dense pages in mmap-resident,
// and read_anonymous has just copied them into anon buffers and rebound every tensor — so the file-
// backed pages are referenced by nobody. Left alone they sit resident until reclaim, doubling the
// dense footprint at the worst moment (right before prefill). Drop them with MADV_DONTNEED: a clean
// read-only mapping, so nothing is lost and nothing will refault the range. Best-effort — needs
// /proc/self/maps to turn a file offset into an address; where that is unreadable the pages stay.
void DenseWeights::drop_mmap_copies(size_t page) {
    resolve_vmas();
    uint64_t dropped = 0;
    for (const DenseTensorRef & d : tensors_) {
        if (!d.tensor || d.size == 0) continue;
        if (is_routed_expert_name(d.tensor->name)) continue;
        const char * a = addr_of(d.file_idx, d.file_off);
        if (!a) continue;
        // Live mmap backing: tensor->data still points into this file range. DONTNEED would
        // free the only copy of those weights (the AnonGpu-frees-experts bug).
        if (d.tensor->data) {
            const char * data = (const char *) d.tensor->data;
            if (data >= a && data < a + d.size) continue;
        }
        // Align INWARD to whole pages (start up, end down), so a page shared with an adjacent tensor
        // that stays mmap-resident — an expert slice, or the next dense tensor — is never dropped.
        uintptr_t a0 = ((uintptr_t) a + page - 1) & ~(uintptr_t) (page - 1);
        uintptr_t a1 = ((uintptr_t) a + d.size) & ~(uintptr_t) (page - 1);
        if (a1 > a0) {
            pio::vm_drop_file_pages((void *) a0, (size_t) (a1 - a0));
            dropped += a1 - a0;
        }
    }
    if (dropped)
        std::fprintf(stderr, "bmoe: dense-weights=%s — dropped %llu MiB of now-unused mmap pages\n",
                     dense_weights_flag(mode_), (unsigned long long) (dropped >> 20));
}

// ── Warmed: one sequential buffered sweep over the dense ranges to populate the page cache ──
void DenseWeights::warm() {
    const size_t chunk = 8ull << 20;
    void * buf = pio::alloc_aligned(align_, chunk);
    if (!buf) return;
    const auto t0 = clock_t_::now();
    uint64_t warmed = 0;
    bool all_ok = true; // sticky, for the report only: a failed shard must not abort the others
    for (size_t s = 0; s < paths_.size() && s < ranges_.size(); ++s) {
        pio::fd_t fd = pio::open_read(paths_[s].c_str(), false);
        if (!pio::fd_ok(fd)) {
            all_ok = false; // best-effort: that shard's pages just stay cold
            continue;
        }
        bool shard_ok = true;
        for (const auto & r : ranges_[s]) {
            for (uint64_t a = r.first; a < r.second && shard_ok;) {
                const long long got = pio::pread_at(fd, buf, (size_t) std::min<uint64_t>(chunk, r.second - a), a);
                if (got <= 0) {
                    shard_ok = all_ok = false;
                    break;
                }
                a += (uint64_t) got;
                warmed += (uint64_t) got;
            }
        }
        pio::close_fd(fd);
    }
    const bool ok = all_ok;
    pio::aligned_free(buf);
    const double s = std::chrono::duration<double>(clock_t_::now() - t0).count();
    std::fprintf(stderr, "bmoe: dense warm-up — %llu MiB in %.1f s%s\n", (unsigned long long) (warmed >> 20), s,
                 ok ? "" : " (partial)");
}

// ── residency sensor ─────────────────────────────────────────────────────────────────
void DenseWeights::sample_residency(size_t page) {
    if (dense_mode_host_copy(mode_))
        sample_anon(page);
    else
        sample_mmap(page);
}

// Anonymous/Pinned: mincore our own buffers directly. Anon memory is reclaimed to zram, and mincore
// reports resident anon pages just as it does file pages, so resident_frac keeps its meaning under
// the flag. Under Pinned the fraction is the EXPERIMENT rather than a diagnostic: dma-buf pages are
// supposed to be reclaim-exempt, so anything below 1.0 falsifies the premise. Should mincore not
// report on a dma-buf mapping at all, vm_resident_sample leaves the counters alone and the fraction
// stays -1 — unmeasured, which must not be read as "nothing is resident".
void DenseWeights::sample_anon(size_t page) {
    if (bases_.empty()) return;
    uint64_t total = 0;
    for (size_t sz : buf_sz_)
        total += sz;
    if (total == 0) return;
    size_t sampled = 0, resident = 0;
    for (int k = 0; k < sample_pages; ++k) {
        uint64_t target = (total * (uint64_t) k) / (uint64_t) sample_pages;
        for (size_t i = 0; i < bases_.size(); ++i) {
            if (target < buf_sz_[i]) {
                const char * a = (const char *) bases_[i] + target;
                const char * pg = (const char *) ((uintptr_t) a & ~(uintptr_t) (page - 1));
                pio::vm_resident_sample(pg, page, &sampled, &resident);
                break;
            }
            target -= buf_sz_[i];
        }
    }
    resident_frac_ = sampled ? (double) resident / (double) sampled : -1.0;
}

// Mmap/Warmed: the weights are llama.cpp's mmaps of the gguf shards; a dense (shard, offset) becomes
// an address through the VMA that backs it (/proc/self/maps), which an app may read, being its own.
// Probe sample_pages points spread evenly across the dense bytes of ALL shards, one page each, so the
// fraction keeps meaning "of the whole dense set" whatever the shard layout.
void DenseWeights::sample_mmap(size_t page) {
    if (ranges_.empty()) return;
    resolve_vmas();

    uint64_t total = 0;
    for (const auto & shard : ranges_)
        for (const auto & r : shard)
            total += r.second - r.first;
    if (total == 0) return;

    size_t sampled = 0, resident = 0;
    for (int k = 0; k < sample_pages; ++k) {
        uint64_t target = (total * (uint64_t) k) / (uint64_t) sample_pages;
        int fi = -1; // found shard; a dense offset of 0 is valid, so the sentinel is the index
        uint64_t off = 0;
        for (size_t s = 0; s < ranges_.size() && fi < 0; ++s) {
            for (const auto & r : ranges_[s]) {
                const uint64_t len = r.second - r.first;
                if (target < len) {
                    fi = (int) s;
                    off = r.first + target;
                    break;
                }
                target -= len;
            }
        }
        // Align the probe DOWN to its page: vm_resident_sample counts only pages fully inside the
        // range, so a single page handed in at an arbitrary offset would clip to nothing.
        if (const char * a = fi >= 0 ? addr_of(fi, off) : nullptr) {
            const char * pg = (const char *) ((uintptr_t) a & ~(uintptr_t) (page - 1));
            pio::vm_resident_sample(pg, page, &sampled, &resident);
        }
    }
    resident_frac_ = sampled ? (double) resident / (double) sampled : -1.0;
}

void DenseWeights::shutdown() {
    for (void * b : bufs_)
        if (b) pio::aligned_free(b);
    for (pio::PinnedAlloc & p : pinned_)
        pio::pinned_free(&p);
    bufs_.clear();
    pinned_.clear();
    bases_.clear();
    buf_sz_.clear();
    tensors_.clear();
    readers_.clear();
}

} // namespace bmoe
