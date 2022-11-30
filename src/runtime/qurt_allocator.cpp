#include "HalideRuntime.h"

extern "C" {

extern void *malloc(size_t);
extern void free(void *);
}

namespace Halide {
namespace Runtime {
namespace Internal {

WEAK void *aligned_malloc(size_t alignment, size_t user_size) {
    // Always round allocations up to alignment size,
    // so that all allocators follow the behavior of aligned_alloc() and
    // return aligned pointer *and* aligned length.
    const size_t aligned_size = align_up(user_size, alignment);

    // We can save a bit of space by special-casing allocations < alignment
    // in size, which we can always fit entirely into 2*alignment.
    const size_t requested_size = (aligned_size <= alignment) ?
                                      (alignment * 2) :
                                      (aligned_size + sizeof(void *) + alignment - 1);

    // malloc() and friends must return a pointer aligned to at least
    // alignof(std::max_align_t); we can't reasonably check that in
    // the runtime, but we can realistically assume it's at least
    // 8-aligned.
    void *orig = malloc(requested_size);
    if (orig == nullptr) {
        // Will result in a failed assertion and a call to halide_error
        return nullptr;
    }

    // We want to store the original pointer prior to the pointer we return.
    void *ptr = (void *)align_up((uintptr_t)orig + sizeof(void *), alignment);
    ((void **)ptr)[-1] = orig;
    return ptr;
}

WEAK void aligned_free(void *ptr) {
    if (ptr) {
        free(((void **)ptr)[-1]);
    }
}

// We keep a small pool of small pre-allocated buffers for use by Halide
// code; some kernels end up doing per-scanline allocations and frees,
// which can cause a noticable performance impact on some workloads.
// 'num_buffers' is the number of pre-allocated buffers and 'buffer_size' is
// the size of each buffer. The pre-allocated buffers are shared among threads
// and we use __sync_val_compare_and_swap primitive to synchronize the buffer
// allocation.
// TODO(psuriana): make num_buffers configurable by user
static const int num_buffers = 10;
static const int buffer_size = 1024 * 64;

WEAK int buf_is_used[num_buffers];
WEAK void *mem_buf[num_buffers] = {
    nullptr,
};

WEAK __attribute__((destructor)) void halide_allocator_cleanup() {
    for (void *buf : mem_buf) {
        aligned_free(buf);
    }
}

}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

WEAK void *halide_default_malloc(void *user_context, size_t x) {
    // Hexagon needs up to 128 byte alignment.
    const size_t alignment = 128;

    if (x <= buffer_size) {
        for (int i = 0; i < num_buffers; ++i) {
            if (__sync_val_compare_and_swap(buf_is_used + i, 0, 1) == 0) {
                if (mem_buf[i] == nullptr) {
                    mem_buf[i] = aligned_malloc(alignment, buffer_size);
                }
                return mem_buf[i];
            }
        }
    }

    return aligned_malloc(alignment, x);
}

WEAK void halide_default_free(void *user_context, void *ptr) {
    for (int i = 0; i < num_buffers; ++i) {
        if (mem_buf[i] == ptr) {
            buf_is_used[i] = 0;
            return;
        }
    }

    aligned_free(ptr);
}

namespace Halide {
namespace Runtime {
namespace Internal {

WEAK halide_malloc_t custom_malloc = halide_default_malloc;
WEAK halide_free_t custom_free = halide_default_free;

}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

extern "C" {

WEAK halide_malloc_t halide_set_custom_malloc(halide_malloc_t user_malloc) {
    // See TODO below.
    halide_print(nullptr, "custom allocators not supported on Hexagon.\n");
    halide_malloc_t result = custom_malloc;
    custom_malloc = user_malloc;
    return result;
}

WEAK halide_free_t halide_set_custom_free(halide_free_t user_free) {
    // See TODO below.
    halide_print(nullptr, "custom allocators not supported on Hexagon.\n");
    halide_free_t result = custom_free;
    custom_free = user_free;
    return result;
}

// TODO: These should be calling custom_malloc/custom_free, but globals are not
// initialized correctly when using mmap_dlopen. We need to fix this, then we
// can enable the custom allocators.
WEAK void *halide_malloc(void *user_context, size_t x) {
    return halide_default_malloc(user_context, x);
}

WEAK void halide_free(void *user_context, void *ptr) {
    halide_default_free(user_context, ptr);
}
}
