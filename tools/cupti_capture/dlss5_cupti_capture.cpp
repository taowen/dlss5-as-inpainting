// Minimal CUPTI module/launch capture for the DLSS5 native carrier.
//
// Loaded through CUDA_INJECTION64_PATH. It deliberately captures metadata and
// the JIT CUBIN only; it does not alter the target process or GPU instructions.
// The CUBIN is copied during the module callback because CUPTI documents the
// resource descriptor as valid only for that callback invocation.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace {

// Keep this probe independent of a full CUDA Toolkit install. These are the
// public CUPTI ABI records needed for resource and driver callbacks; the
// standalone CUPTI wheel supplies the implementation DLL at runtime.
#define CUPTIAPI __stdcall
using CUptiResult = int;
using CUpti_CallbackDomain = uint32_t;
using CUpti_CallbackId = uint32_t;
using CUcontext = void*;
using CUstream = void*;
struct CUpti_Subscriber_st;
using CUpti_SubscriberHandle = CUpti_Subscriber_st*;
enum : CUptiResult { CUPTI_SUCCESS = 0 };
enum : CUpti_CallbackDomain {
    CUPTI_CB_DOMAIN_DRIVER_API = 1,
    CUPTI_CB_DOMAIN_RESOURCE = 3,
};
enum : CUpti_CallbackId {
    CUPTI_CBID_RESOURCE_MODULE_LOADED = 6,
    CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING = 7,
    CUPTI_DRIVER_TRACE_CBID_cuLaunch = 115,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel = 307,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz = 442,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx = 652,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz = 653,
};
enum : uint32_t {
    CUPTI_API_ENTER = 0,
    CUPTI_API_EXIT = 1,
};
struct CUpti_ResourceData {
    CUcontext context;
    union { CUstream stream; } resourceHandle;
    void* resourceDescriptor;
};
struct CUpti_ModuleResourceData {
    uint32_t moduleId;
    size_t cubinSize;
    const char* pCubin;
};
struct CUpti_CallbackData {
    uint32_t callbackSite;
    const char* functionName;
    const void* functionParams;
    void* functionReturnValue;
    const char* symbolName;
    CUcontext context;
    uint32_t contextUid;
    uint64_t* correlationData;
    uint32_t correlationId;
};
using CUpti_CallbackFunc = void(CUPTIAPI*)(void*, CUpti_CallbackDomain,
                                            CUpti_CallbackId, const void*);

using CuptiSubscribe = CUptiResult(CUPTIAPI*)(
    CUpti_SubscriberHandle*, CUpti_CallbackFunc, void*);
using CuptiEnableCallback = CUptiResult(CUPTIAPI*)(
    uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);

HMODULE g_cupti = nullptr;
CUpti_SubscriberHandle g_subscriber = nullptr;
CuptiSubscribe g_subscribe = nullptr;
CuptiEnableCallback g_enable_callback = nullptr;
std::filesystem::path g_output;
std::ofstream g_events;
std::mutex g_mutex;
std::once_flag g_once;
std::atomic<uint32_t> g_module_count{0};

std::string GetEnv(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

std::filesystem::path OutputDirectory() {
    const std::string configured = GetEnv("DLSS5_CUPTI_CAPTURE_DIR");
    if (!configured.empty()) return std::filesystem::path(configured);
    wchar_t current[MAX_PATH] = {};
    const DWORD length = GetCurrentDirectoryW(MAX_PATH, current);
    if (length != 0 && length < MAX_PATH) return std::filesystem::path(current);
    return std::filesystem::temp_directory_path() / "dlss5-cupti-capture";
}

std::string JsonString(const char* value) {
    std::string result = "\"";
    if (!value) return result + "\"";
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
        switch (*p) {
            case '\\': result += "\\\\"; break;
            case '\"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result.push_back(static_cast<char>(*p)); break;
        }
    }
    result += '"';
    return result;
}

void Record(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_events) {
        g_events << line << '\n';
        g_events.flush();
    }
}

void CaptureModule(const CUpti_ResourceData* resource) {
    if (!resource || !resource->resourceDescriptor) return;
    const auto* module = static_cast<const CUpti_ModuleResourceData*>(
        resource->resourceDescriptor);
    if (!module || !module->pCubin || module->cubinSize == 0) return;

    const std::filesystem::path cubin_path =
        g_output / ("module_" + std::to_string(module->moduleId) + ".cubin");
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        std::ofstream cubin(cubin_path, std::ios::binary | std::ios::trunc);
        if (cubin) {
            cubin.write(module->pCubin, static_cast<std::streamsize>(module->cubinSize));
        }
    }
    g_module_count.fetch_add(1, std::memory_order_relaxed);
    Record("{\"event\":\"module_loaded\",\"module_id\":" +
           std::to_string(module->moduleId) + ",\"cubin_size\":" +
           std::to_string(module->cubinSize) + ",\"cubin\":" +
           JsonString(cubin_path.filename().string().c_str()) + "}");
}

void CUPTIAPI Callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId callback_id,
                       const void* callback_data) {
    if (domain == CUPTI_CB_DOMAIN_RESOURCE &&
        (callback_id == CUPTI_CBID_RESOURCE_MODULE_LOADED ||
         callback_id == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING)) {
        CaptureModule(static_cast<const CUpti_ResourceData*>(callback_data));
        return;
    }
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API ||
        (callback_id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel &&
         callback_id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz &&
         callback_id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx &&
         callback_id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz &&
         callback_id != CUPTI_DRIVER_TRACE_CBID_cuLaunch)) {
        return;
    }
    const auto* callback_info = static_cast<const CUpti_CallbackData*>(callback_data);
    if (!callback_info || callback_info->callbackSite != CUPTI_API_ENTER) return;
    Record("{\"event\":\"kernel_launch\",\"name\":" +
           JsonString(callback_info->functionName) + ",\"correlation_id\":" +
           std::to_string(callback_info->correlationId) + "}");
}

void Initialise() {
    if (GetEnv("DLSS5_CUPTI_NOOP") == "1") return;
    const std::string configured_dll = GetEnv("DLSS5_CUPTI_DLL");
    if (!configured_dll.empty()) {
        g_cupti = LoadLibraryA(configured_dll.c_str());
    } else {
        g_cupti = LoadLibraryA("cupti64_2026.2.1.dll");
    }
    if (!g_cupti) return;
    g_subscribe = reinterpret_cast<CuptiSubscribe>(GetProcAddress(g_cupti, "cuptiSubscribe"));
    g_enable_callback = reinterpret_cast<CuptiEnableCallback>(
        GetProcAddress(g_cupti, "cuptiEnableCallback"));
    if (!g_subscribe || !g_enable_callback) return;

    g_output = OutputDirectory();
    std::error_code error;
    std::filesystem::create_directories(g_output, error);
    g_events.open(g_output / "events.jsonl", std::ios::out | std::ios::trunc);
    Record("{\"event\":\"capture_initialized\",\"output\":" +
           JsonString(g_output.string().c_str()) + "}");

    const CUptiResult subscribed = g_subscribe(&g_subscriber, Callback, nullptr);
    Record("{\"event\":\"cupti_subscribe\",\"result\":" +
           std::to_string(subscribed) + "}");
    if (subscribed != CUPTI_SUCCESS) return;
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                      CUPTI_CBID_RESOURCE_MODULE_LOADED);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                      CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz);
    g_enable_callback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunch);
}

}  // namespace

extern "C" __declspec(dllexport) int InitializeInjection(void) {
    std::call_once(g_once, Initialise);
    // CUDA's injection loader treats a zero return as an injection failure;
    // that can poison unrelated D3D12 work before CUPTI is initialized. The
    // capture is best-effort, so acknowledge the injection after attempting
    // setup and let the target continue even when CUPTI rejects a subscriber.
    return 1;
}
