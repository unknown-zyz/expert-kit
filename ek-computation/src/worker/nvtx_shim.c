#include <stdint.h>

#ifdef EK_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>

void ek_nvtx_range_push(const char *message) { nvtxRangePushA(message); }
void ek_nvtx_range_pop(void) { nvtxRangePop(); }
uint64_t ek_nvtx_range_start(const char *message) { return nvtxRangeStartA(message); }
void ek_nvtx_range_end(uint64_t id) { nvtxRangeEnd(id); }
#else
void ek_nvtx_range_push(const char *message) { (void)message; }
void ek_nvtx_range_pop(void) {}
uint64_t ek_nvtx_range_start(const char *message) { (void)message; return 0; }
void ek_nvtx_range_end(uint64_t id) { (void)id; }
#endif
