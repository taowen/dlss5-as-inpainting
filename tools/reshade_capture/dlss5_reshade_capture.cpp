// Capture the graphics-side ABI around NGX/DLSS5 without modifying commands.
// This intentionally logs handles, dimensions, layouts, push constants and
// dispatches only. It is a diagnostic add-on for a disposable ReShade runtime.

#include <windows.h>
#include <d3d12.h>
#include <MinHook.h>
#include <reshade.hpp>

#include <cstdint>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <cstring>
#include <memory>
#include <vector>

using namespace reshade::api;

namespace {

std::mutex g_mutex;
std::ofstream g_log;
std::once_flag g_hooks_once;

using CUresult = unsigned int;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CuModuleLoadDataFn = CUresult(__stdcall *)(CUmodule *, const void *);
using CuModuleLoadDataExFn = CUresult(__stdcall *)(CUmodule *, const void *, unsigned int, void *, void *);
using CuModuleGetFunctionFn = CUresult(__stdcall *)(CUfunction *, CUmodule, const char *);
using CuGetExportTableFn = CUresult(__stdcall *)(const void **, const void *);
using CuLaunchKernelFn = CUresult(__stdcall *)(CUfunction, unsigned int, unsigned int, unsigned int,
                                                unsigned int, unsigned int, unsigned int, unsigned int,
                                                CUstream, void **, void **);

CuModuleLoadDataFn g_cu_module_load_data = nullptr;
CuModuleLoadDataExFn g_cu_module_load_data_ex = nullptr;
CuModuleGetFunctionFn g_cu_module_get_function = nullptr;
CuLaunchKernelFn g_cu_launch_kernel = nullptr;
CuGetExportTableFn g_cu_get_export_table = nullptr;
unsigned int g_cuda_module_index = 0;

using D3D12DispatchFn = void(STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, UINT, UINT, UINT);
using D3D12SetPipelineStateFn = void(STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12PipelineState *);
using D3D12SetComputeRootSignatureFn = void(STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, ID3D12RootSignature *);
using D3D12SetComputeRootDescriptorTableFn = void(STDMETHODCALLTYPE *)(ID3D12GraphicsCommandList *, UINT, D3D12_GPU_DESCRIPTOR_HANDLE);
using D3D12CreateComputePipelineStateFn = HRESULT(STDMETHODCALLTYPE *)(
    ID3D12Device *, const D3D12_COMPUTE_PIPELINE_STATE_DESC *, REFIID, void **);
using D3D12CreateDescriptorHeapFn = HRESULT(STDMETHODCALLTYPE *)(
    ID3D12Device *, const D3D12_DESCRIPTOR_HEAP_DESC *, REFIID, void **);
using D3D12CreateShaderResourceViewFn = void(STDMETHODCALLTYPE *)(
    ID3D12Device *, ID3D12Resource *, const D3D12_SHADER_RESOURCE_VIEW_DESC *, D3D12_CPU_DESCRIPTOR_HANDLE);
using D3D12CreateUnorderedAccessViewFn = void(STDMETHODCALLTYPE *)(
    ID3D12Device *, ID3D12Resource *, ID3D12Resource *, const D3D12_UNORDERED_ACCESS_VIEW_DESC *, D3D12_CPU_DESCRIPTOR_HANDLE);
D3D12DispatchFn g_d3d12_dispatch = nullptr;
D3D12SetPipelineStateFn g_d3d12_set_pipeline_state = nullptr;
D3D12SetComputeRootSignatureFn g_d3d12_set_compute_root_signature = nullptr;
D3D12SetComputeRootDescriptorTableFn g_d3d12_set_compute_root_descriptor_table = nullptr;
D3D12CreateComputePipelineStateFn g_d3d12_create_compute_pipeline_state = nullptr;
D3D12CreateDescriptorHeapFn g_d3d12_create_descriptor_heap = nullptr;
D3D12CreateShaderResourceViewFn g_d3d12_create_shader_resource_view = nullptr;
D3D12CreateUnorderedAccessViewFn g_d3d12_create_unordered_access_view = nullptr;
unsigned int g_d3d12_pso_index = 0;
struct DarkTableProxy {
    void *original = nullptr;
    void *replacement = nullptr;
    size_t bytes = 0;
    std::vector<void *> thunks;
};
struct DarkCallRecord {
    volatile uint64_t count = 0;
    uintptr_t register_args[4]{};
    uintptr_t stack_args[4]{};
    const void *table = nullptr;
    size_t index = 0;
    void *original = nullptr;
    uintptr_t nonvolatile_args[8]{}; // rbx, rbp, rsi, rdi, r12-r15
    uintptr_t extra_args[2]{};       // r10, r11
    uintptr_t return_address = 0;
    uintptr_t last_register_args[4]{};
    uintptr_t last_stack_args[4]{};
    uintptr_t last_nonvolatile_args[8]{};
    uintptr_t last_extra_args[2]{};
    uintptr_t last_return_address = 0;
};
static_assert(sizeof(DarkCallRecord) == 0x150, "unexpected dark call record layout");

struct DarkTraceHeader {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint32_t pid = 0;
    uint32_t max_records = 0;
    uint64_t record_bytes = 0;
    uint64_t record_count = 0;
    uint64_t reserved[3]{};
};
static_assert(sizeof(DarkTraceHeader) == 64, "unexpected dark trace header layout");

constexpr uint64_t kDarkTraceMagic = 0x4452434535444c53ull; // "SLD5ECRD"
constexpr uint32_t kDarkTraceVersion = 1;
constexpr size_t kDarkTraceMaxRecords = 1024;
constexpr size_t kDarkTraceBytes = sizeof(DarkTraceHeader) +
                                    kDarkTraceMaxRecords * sizeof(DarkCallRecord);
std::mutex g_dark_mutex;
std::vector<DarkTableProxy> g_dark_tables;
std::vector<DarkCallRecord *> g_dark_records;
std::vector<std::unique_ptr<DarkCallRecord>> g_dark_owned_records;
HANDLE g_dark_trace_file = INVALID_HANDLE_VALUE;
HANDLE g_dark_trace_mapping = nullptr;
void *g_dark_trace_view = nullptr;
DarkTraceHeader *g_dark_trace_header = nullptr;
size_t g_dark_trace_record_cursor = 0;
unsigned int g_dark_blob_index = 0;

using GetProcAddressFn = FARPROC(WINAPI *)(HMODULE, LPCSTR);
using LoadLibraryExWFn = HMODULE(WINAPI *)(LPCWSTR, HANDLE, DWORD);
using LoadLibraryWFn = HMODULE(WINAPI *)(LPCWSTR);
GetProcAddressFn g_get_proc_address = nullptr;
LoadLibraryExWFn g_load_library_ex_w = nullptr;
LoadLibraryWFn g_load_library_w = nullptr;

std::string hex_bytes(const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes; ++i) out << std::setw(2) << unsigned(p[i]);
    return out.str();
}

void open_log() {
    wchar_t path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring full(path, n);
    const size_t slash = full.find_last_of(L"\\/");
    if (slash != std::wstring::npos) full.resize(slash + 1);
    full += L"dlss5_reshade_capture.log";
    g_log.open(full, std::ios::out | std::ios::app);
}

template <typename F>
void log(F &&make_line) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_log.is_open()) open_log();
    g_log << make_line() << '\n';
    g_log.flush();
}

std::wstring module_name(HMODULE module) {
    wchar_t path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
    std::wstring result(path, n);
    const size_t slash = result.find_last_of(L"\\/");
    if (slash != std::wstring::npos) result.erase(0, slash + 1);
    return result;
}

std::string narrow(const wchar_t *text) {
    if (!text) return "<null>";
    char buffer[512]{};
    const int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, buffer, sizeof(buffer), nullptr, nullptr);
    return n > 0 ? std::string(buffer, n - 1) : std::string("<unprintable>");
}

std::string address_module(uintptr_t address) {
    if (!address) return "<null>";
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &module)) return "<unknown>";
    return narrow(module_name(module).c_str());
}

uintptr_t address_module_base(uintptr_t address) {
    if (!address) return 0;
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &module)) return 0;
    return reinterpret_cast<uintptr_t>(module);
}

size_t guess_elf_size(const void *image) {
    if (!image) return 0;
    MEMORY_BASIC_INFORMATION region{};
    if (VirtualQuery(image, &region, sizeof(region)) != sizeof(region) ||
        region.State != MEM_COMMIT || region.Protect == PAGE_NOACCESS ||
        region.Protect == PAGE_GUARD ||
        reinterpret_cast<uintptr_t>(image) + 0x40 < reinterpret_cast<uintptr_t>(image) ||
        reinterpret_cast<uintptr_t>(image) + 0x40 >=
            reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize) return 0;
    const auto *p = static_cast<const uint8_t *>(image);
    if (p[0] != 0x7f || p[1] != 'E' || p[2] != 'L' || p[3] != 'F' || p[4] != 2) return 0;
    const auto read16 = [&](size_t offset) -> uint16_t {
        uint16_t value; std::memcpy(&value, p + offset, sizeof(value)); return value;
    };
    const auto read64 = [&](size_t offset) -> uint64_t {
        uint64_t value; std::memcpy(&value, p + offset, sizeof(value)); return value;
    };
    const uint64_t section_offset = read64(0x28);
    const uint16_t section_entry_size = read16(0x3a);
    const uint16_t section_count = read16(0x3c);
    if (section_entry_size < 64 || section_count == 0 || section_count > 4096 || section_offset > 0x4000000) return 0;
    uint64_t size = 0x40;
    for (uint16_t i = 0; i < section_count; ++i) {
        const uint64_t entry = section_offset + uint64_t(i) * section_entry_size;
        if (entry > 0x4000000 ||
            !VirtualQuery(p + entry, &region, sizeof(region)) ||
            region.State != MEM_COMMIT || region.Protect == PAGE_NOACCESS ||
            region.Protect == PAGE_GUARD ||
            entry + 0x28 > reinterpret_cast<uintptr_t>(region.BaseAddress) + region.RegionSize -
                reinterpret_cast<uintptr_t>(p)) return 0;
        const uint64_t offset = read64(static_cast<size_t>(entry) + 0x18);
        const uint64_t section_size = read64(static_cast<size_t>(entry) + 0x20);
        if (offset > 0x4000000 || section_size > 0x4000000 || offset + section_size > 0x4000000) return 0;
        if (offset + section_size > size) size = offset + section_size;
    }
    return static_cast<size_t>(size);
}

void dump_cuda_image(const char *api, const void *image) {
    const size_t size = guess_elf_size(image);
    log([&] {
        std::ostringstream s;
        s << api << " image=0x" << std::hex << reinterpret_cast<uintptr_t>(image)
          << " guessed_bytes=" << std::dec << size;
        if (image) s << " prefix=" << hex_bytes(image, 16);
        return s.str();
    });
    if (size == 0 || size > 0x4000000 || !image) return;
    std::ostringstream name;
    name << "dlss5_cuda_module_" << g_cuda_module_index++ << ".cubin";
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_log.is_open()) open_log();
    wchar_t executable_path[MAX_PATH]{};
    const DWORD executable_length = GetModuleFileNameW(nullptr, executable_path, MAX_PATH);
    std::wstring full(executable_path, executable_length);
    const size_t slash = full.find_last_of(L"\\/");
    if (slash != std::wstring::npos) full.resize(slash + 1);
    const std::string file_name = name.str();
    full += std::wstring(file_name.begin(), file_name.end());
    std::ofstream out(full, std::ios::binary | std::ios::trunc);
    if (out) out.write(static_cast<const char *>(image), static_cast<std::streamsize>(size));
}

void dump_dark_memory(const char *api, uintptr_t value) {
    if (value < 0x10000) return;
    uint8_t bytes[128]{};
    SIZE_T copied = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void *>(value),
                           bytes, sizeof(bytes), &copied) || copied == 0) return;
    log([&] {
        std::ostringstream s;
        s << api << " ptr=0x" << std::hex << value << " bytes=" << std::dec << copied
          << " data=" << hex_bytes(bytes, copied);
        return s.str();
    });
}

void dump_dark_binary(const char *api, uintptr_t value) {
    if (value < 0x10000) return;

    const uint8_t elf_magic[4] = {0x7f, 'E', 'L', 'F'};
    const uint8_t bundle_magic[4] = {0x50, 0xed, 0x55, 0xba};
    uint8_t prefix[0x100]{};
    SIZE_T copied = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void *>(value),
                           prefix, sizeof(prefix), &copied) || copied < 4) return;

    size_t image_offset = 0;
    size_t blob_size = 0;
    if (std::memcmp(prefix, elf_magic, sizeof(elf_magic)) == 0) {
        image_offset = 0;
    } else if (copied >= 0x54 && std::memcmp(prefix, bundle_magic, sizeof(bundle_magic)) == 0 &&
               std::memcmp(prefix + 0x50, elf_magic, sizeof(elf_magic)) == 0) {
        image_offset = 0x50;
        uint64_t declared_size = 0;
        std::memcpy(&declared_size, prefix + 8, sizeof(declared_size));
        if (declared_size >= image_offset + 4 && declared_size <= 0x4000000)
            blob_size = static_cast<size_t>(declared_size);
    } else {
        return;
    }

    if (blob_size == 0) {
        const size_t image_size = guess_elf_size(reinterpret_cast<const uint8_t *>(value) + image_offset);
        if (image_size == 0 || image_size > 0x4000000 || image_offset + image_size > 0x4000000) return;
        blob_size = image_offset + image_size;
    }
    std::vector<uint8_t> blob(blob_size);
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void *>(value),
                           blob.data(), blob.size(), &copied) || copied != blob.size()) return;

    std::ostringstream name;
    name << "dlss5_dark_blob_" << GetCurrentProcessId() << '_' << g_dark_blob_index++ << ".bin";
    const std::string file_name = name.str();
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_log.is_open()) open_log();
        wchar_t executable_path[MAX_PATH]{};
        const DWORD executable_length = GetModuleFileNameW(nullptr, executable_path, MAX_PATH);
        std::wstring full(executable_path, executable_length);
        const size_t slash = full.find_last_of(L"\\/");
        if (slash != std::wstring::npos) full.resize(slash + 1);
        full += std::wstring(file_name.begin(), file_name.end());
        std::ofstream out(full, std::ios::binary | std::ios::trunc);
        if (out) out.write(reinterpret_cast<const char *>(blob.data()), static_cast<std::streamsize>(blob.size()));
    }
    log([&] {
        std::ostringstream s;
        s << "dark_binary_dump api=" << api << " ptr=0x" << std::hex << value
          << " image_offset=0x" << image_offset << " bytes=" << std::dec << blob.size()
          << " file=" << file_name;
        return s.str();
    });
}

void init_dark_trace() {
    if (g_dark_trace_view) return;

    wchar_t path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return;
    std::wstring full(path, n);
    const size_t slash = full.find_last_of(L"\\/");
    if (slash != std::wstring::npos) full.resize(slash + 1);
    full += L"dlss5_driver_trace_";
    full += std::to_wstring(GetCurrentProcessId());
    full += L".bin";

    g_dark_trace_file = CreateFileW(full.c_str(), GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g_dark_trace_file == INVALID_HANDLE_VALUE) {
        log([&] { return std::string("dark_trace_file_failed error=") + std::to_string(GetLastError()); });
        return;
    }

    LARGE_INTEGER size{};
    size.QuadPart = static_cast<LONGLONG>(kDarkTraceBytes);
    if (!SetFilePointerEx(g_dark_trace_file, size, nullptr, FILE_BEGIN) || !SetEndOfFile(g_dark_trace_file)) {
        log([&] { return std::string("dark_trace_resize_failed error=") + std::to_string(GetLastError()); });
        CloseHandle(g_dark_trace_file);
        g_dark_trace_file = INVALID_HANDLE_VALUE;
        return;
    }

    g_dark_trace_mapping = CreateFileMappingW(g_dark_trace_file, nullptr, PAGE_READWRITE, 0,
                                               static_cast<DWORD>(kDarkTraceBytes), nullptr);
    if (!g_dark_trace_mapping) {
        log([&] { return std::string("dark_trace_mapping_failed error=") + std::to_string(GetLastError()); });
        CloseHandle(g_dark_trace_file);
        g_dark_trace_file = INVALID_HANDLE_VALUE;
        return;
    }
    g_dark_trace_view = MapViewOfFile(g_dark_trace_mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, kDarkTraceBytes);
    if (!g_dark_trace_view) {
        log([&] { return std::string("dark_trace_map_failed error=") + std::to_string(GetLastError()); });
        CloseHandle(g_dark_trace_mapping);
        CloseHandle(g_dark_trace_file);
        g_dark_trace_mapping = nullptr;
        g_dark_trace_file = INVALID_HANDLE_VALUE;
        return;
    }
    std::memset(g_dark_trace_view, 0, kDarkTraceBytes);
    g_dark_trace_header = static_cast<DarkTraceHeader *>(g_dark_trace_view);
    g_dark_trace_header->magic = kDarkTraceMagic;
    g_dark_trace_header->version = kDarkTraceVersion;
    g_dark_trace_header->header_bytes = sizeof(DarkTraceHeader);
    g_dark_trace_header->pid = GetCurrentProcessId();
    g_dark_trace_header->max_records = static_cast<uint32_t>(kDarkTraceMaxRecords);
    g_dark_trace_header->record_bytes = sizeof(DarkCallRecord);
    FlushViewOfFile(g_dark_trace_view, sizeof(DarkTraceHeader));
    log([&] {
        std::ostringstream s;
        s << "dark_trace_file path=" << narrow(full.c_str()) << " bytes=" << std::dec << kDarkTraceBytes;
        return s.str();
    });
}

void close_dark_trace() {
    if (g_dark_trace_view) {
        if (g_dark_trace_header) g_dark_trace_header->record_count = g_dark_trace_record_cursor;
        FlushViewOfFile(g_dark_trace_view, 0);
        UnmapViewOfFile(g_dark_trace_view);
        g_dark_trace_view = nullptr;
        g_dark_trace_header = nullptr;
    }
    if (g_dark_trace_mapping) {
        CloseHandle(g_dark_trace_mapping);
        g_dark_trace_mapping = nullptr;
    }
    if (g_dark_trace_file != INVALID_HANDLE_VALUE) {
        CloseHandle(g_dark_trace_file);
        g_dark_trace_file = INVALID_HANDLE_VALUE;
    }
}

bool write_runtime_binary(const std::string &file_name, const void *data, size_t bytes) {
    if (!data || bytes == 0 || bytes > 0x4000000) return false;
    wchar_t executable_path[MAX_PATH]{};
    const DWORD executable_length = GetModuleFileNameW(nullptr, executable_path, MAX_PATH);
    if (executable_length == 0 || executable_length >= MAX_PATH) return false;
    std::wstring full(executable_path, executable_length);
    const size_t slash = full.find_last_of(L"\\/");
    if (slash != std::wstring::npos) full.resize(slash + 1);
    full += std::wstring(file_name.begin(), file_name.end());
    std::lock_guard<std::mutex> lock(g_mutex);
    std::ofstream out(full, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    return out.good();
}

void STDMETHODCALLTYPE hook_d3d12_dispatch(ID3D12GraphicsCommandList *list, UINT x, UINT y, UINT z) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_dispatch list=0x" << std::hex << reinterpret_cast<uintptr_t>(list)
          << " groups=" << std::dec << x << ',' << y << ',' << z;
        return s.str();
    });
    if (g_d3d12_dispatch) g_d3d12_dispatch(list, x, y, z);
}

void STDMETHODCALLTYPE hook_d3d12_set_pipeline_state(ID3D12GraphicsCommandList *list,
                                                      ID3D12PipelineState *pipeline) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_set_pipeline_state list=0x" << std::hex << reinterpret_cast<uintptr_t>(list)
          << " pipeline=0x" << reinterpret_cast<uintptr_t>(pipeline);
        return s.str();
    });
    if (g_d3d12_set_pipeline_state) g_d3d12_set_pipeline_state(list, pipeline);
}

void STDMETHODCALLTYPE hook_d3d12_set_compute_root_signature(ID3D12GraphicsCommandList *list,
                                                               ID3D12RootSignature *signature) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_set_compute_root_signature list=0x" << std::hex
          << reinterpret_cast<uintptr_t>(list) << " signature=0x"
          << reinterpret_cast<uintptr_t>(signature);
        return s.str();
    });
    if (g_d3d12_set_compute_root_signature) g_d3d12_set_compute_root_signature(list, signature);
}

void STDMETHODCALLTYPE hook_d3d12_set_compute_root_descriptor_table(
    ID3D12GraphicsCommandList *list, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE handle) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_set_compute_root_descriptor_table list=0x" << std::hex
          << reinterpret_cast<uintptr_t>(list) << " index=" << std::dec << index
          << " gpu_handle=0x" << std::hex << handle.ptr;
        return s.str();
    });
    if (g_d3d12_set_compute_root_descriptor_table)
        g_d3d12_set_compute_root_descriptor_table(list, index, handle);
}

HRESULT STDMETHODCALLTYPE hook_d3d12_create_compute_pipeline_state(
    ID3D12Device *device, const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc,
    REFIID riid, void **result) {
    const size_t shader_bytes = desc ? static_cast<size_t>(desc->CS.BytecodeLength) : 0;
    const void *shader = desc ? desc->CS.pShaderBytecode : nullptr;
    const unsigned int index = g_d3d12_pso_index++;
    std::ostringstream name;
    name << "dlss5_d3d12_cs_" << GetCurrentProcessId() << '_' << index << ".dxil";
    const bool dumped = write_runtime_binary(name.str(), shader, shader_bytes);
    const HRESULT hr = g_d3d12_create_compute_pipeline_state
        ? g_d3d12_create_compute_pipeline_state(device, desc, riid, result)
        : E_FAIL;
    log([&] {
        std::ostringstream s;
        s << "d3d12_create_compute_pso device=0x" << std::hex
          << reinterpret_cast<uintptr_t>(device) << " result=0x"
          << static_cast<unsigned long>(hr) << " pso=0x"
          << (result ? reinterpret_cast<uintptr_t>(*result) : 0)
          << " root_signature=0x" << (desc ? reinterpret_cast<uintptr_t>(desc->pRootSignature) : 0)
          << " cs=0x" << reinterpret_cast<uintptr_t>(shader)
          << " cs_bytes=" << std::dec << shader_bytes << " dumped=" << dumped
          << " file=" << name.str();
        return s.str();
    });
    return hr;
}

HRESULT STDMETHODCALLTYPE hook_d3d12_create_descriptor_heap(
    ID3D12Device *device, const D3D12_DESCRIPTOR_HEAP_DESC *desc,
    REFIID riid, void **result) {
    const HRESULT hr = g_d3d12_create_descriptor_heap
        ? g_d3d12_create_descriptor_heap(device, desc, riid, result)
        : E_FAIL;
    if (SUCCEEDED(hr) && result && *result && desc) {
        auto *heap = reinterpret_cast<ID3D12DescriptorHeap *>(*result);
        const D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
        const UINT increment = device->GetDescriptorHandleIncrementSize(desc->Type);
        log([&] {
            std::ostringstream s;
            s << "d3d12_create_descriptor_heap heap=0x" << std::hex
              << reinterpret_cast<uintptr_t>(heap) << " type=" << std::dec << uint32_t(desc->Type)
              << " flags=0x" << uint32_t(desc->Flags) << " descriptors=" << desc->NumDescriptors
              << " increment=" << increment << " cpu_start=0x" << std::hex << cpu.ptr
              << " gpu_start=0x" << gpu.ptr;
            return s.str();
        });
    }
    return hr;
}

void STDMETHODCALLTYPE hook_d3d12_create_shader_resource_view(
    ID3D12Device *device, ID3D12Resource *resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_create_srv resource=0x" << std::hex << reinterpret_cast<uintptr_t>(resource)
          << " cpu_handle=0x" << handle.ptr;
        if (desc) s << " format=" << std::dec << uint32_t(desc->Format)
                    << " dimension=" << uint32_t(desc->ViewDimension);
        return s.str();
    });
    if (g_d3d12_create_shader_resource_view)
        g_d3d12_create_shader_resource_view(device, resource, desc, handle);
}

void STDMETHODCALLTYPE hook_d3d12_create_unordered_access_view(
    ID3D12Device *device, ID3D12Resource *resource, ID3D12Resource *counter,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    log([&] {
        std::ostringstream s;
        s << "d3d12_create_uav resource=0x" << std::hex << reinterpret_cast<uintptr_t>(resource)
          << " counter=0x" << reinterpret_cast<uintptr_t>(counter)
          << " cpu_handle=0x" << handle.ptr;
        if (desc) s << " format=" << std::dec << uint32_t(desc->Format)
                    << " dimension=" << uint32_t(desc->ViewDimension);
        return s.str();
    });
    if (g_d3d12_create_unordered_access_view)
        g_d3d12_create_unordered_access_view(device, resource, counter, desc, handle);
}

void install_d3d12_command_list_hooks(command_list *cmd_list) {
    if (!cmd_list || cmd_list->get_device()->get_api() != device_api::d3d12) return;
    auto *native = reinterpret_cast<ID3D12GraphicsCommandList *>(cmd_list->get_native());
    if (!native) return;
    auto **vtable = *reinterpret_cast<void ***>(native);
    struct Method {
        size_t index;
        const char *name;
        void *replacement;
        void **original;
    } methods[] = {
        {14, "Dispatch", reinterpret_cast<void *>(&hook_d3d12_dispatch),
         reinterpret_cast<void **>(&g_d3d12_dispatch)},
        {25, "SetPipelineState", reinterpret_cast<void *>(&hook_d3d12_set_pipeline_state),
         reinterpret_cast<void **>(&g_d3d12_set_pipeline_state)},
        {29, "SetComputeRootSignature", reinterpret_cast<void *>(&hook_d3d12_set_compute_root_signature),
         reinterpret_cast<void **>(&g_d3d12_set_compute_root_signature)},
        {31, "SetComputeRootDescriptorTable",
         reinterpret_cast<void *>(&hook_d3d12_set_compute_root_descriptor_table),
         reinterpret_cast<void **>(&g_d3d12_set_compute_root_descriptor_table)},
    };
    for (const Method &method : methods) {
        if (!vtable[method.index]) continue;
        const MH_STATUS status = MH_CreateHook(vtable[method.index], method.replacement,
                                               reinterpret_cast<LPVOID *>(method.original));
        if (status == MH_OK || status == MH_ERROR_ALREADY_CREATED) {
            MH_EnableHook(vtable[method.index]);
            log([&] {
                std::ostringstream s;
                s << "d3d12_hook_installed name=" << method.name << " target=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(vtable[method.index]);
                return s.str();
            });
        } else {
            log([&] {
                std::ostringstream s;
                s << "d3d12_hook_failed name=" << method.name << " status=" << std::dec << int(status);
                return s.str();
            });
        }
    }
}

void install_d3d12_device_hooks(device *dev) {
    if (!dev || dev->get_api() != device_api::d3d12) return;
    auto *native = reinterpret_cast<ID3D12Device *>(dev->get_native());
    if (!native) return;
    auto **vtable = *reinterpret_cast<void ***>(native);
    struct Method {
        size_t index;
        const char *name;
        void *replacement;
        void **original;
    } methods[] = {
        {11, "CreateComputePipelineState", reinterpret_cast<void *>(&hook_d3d12_create_compute_pipeline_state),
         reinterpret_cast<void **>(&g_d3d12_create_compute_pipeline_state)},
        {14, "CreateDescriptorHeap", reinterpret_cast<void *>(&hook_d3d12_create_descriptor_heap),
         reinterpret_cast<void **>(&g_d3d12_create_descriptor_heap)},
        {18, "CreateShaderResourceView", reinterpret_cast<void *>(&hook_d3d12_create_shader_resource_view),
         reinterpret_cast<void **>(&g_d3d12_create_shader_resource_view)},
        {19, "CreateUnorderedAccessView", reinterpret_cast<void *>(&hook_d3d12_create_unordered_access_view),
         reinterpret_cast<void **>(&g_d3d12_create_unordered_access_view)},
    };
    for (const Method &method : methods) {
        if (!vtable[method.index]) continue;
        const MH_STATUS status = MH_CreateHook(vtable[method.index], method.replacement,
                                               reinterpret_cast<LPVOID *>(method.original));
        if (status == MH_OK || status == MH_ERROR_ALREADY_CREATED) {
            MH_EnableHook(vtable[method.index]);
            log([&] {
                std::ostringstream s;
                s << "d3d12_hook_installed name=" << method.name << " target=0x" << std::hex
                  << reinterpret_cast<uintptr_t>(vtable[method.index]);
                return s.str();
            });
        } else {
            log([&] {
                std::ostringstream s;
                s << "d3d12_hook_failed name=" << method.name << " status=" << std::dec << int(status);
                return s.str();
            });
        }
    }
}

void emit_u64(std::vector<uint8_t> &code, uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) code.push_back(static_cast<uint8_t>(value >> (i * 8)));
}

void *make_dark_thunk(void *original, DarkCallRecord *record) {
    char no_op[4]{};
    if (GetEnvironmentVariableA("DLSS5_DARK_NOOP", no_op, sizeof(no_op)) > 0) {
        std::vector<uint8_t> direct = {0x48, 0xb8};
        emit_u64(direct, reinterpret_cast<uintptr_t>(original));
        direct.insert(direct.end(), {0xff, 0xe0});
        void *memory = VirtualAlloc(nullptr, direct.size(), MEM_COMMIT | MEM_RESERVE,
                                    PAGE_EXECUTE_READWRITE);
        if (!memory) return nullptr;
        std::memcpy(memory, direct.data(), direct.size());
        FlushInstructionCache(GetCurrentProcess(), memory, direct.size());
        return memory;
    }
    // Do not call into C++ from this thunk. These private CUDA table entries
    // are not documented C ABI functions and can carry arguments outside the
    // ordinary driver prototypes. Record the first register/stack words with
    // volatile scratch registers, restore them, and tail-jump unchanged.
    std::vector<uint8_t> code = {
        0x50,                                      // push rax
        0x41, 0x52,                                // push r10
        0x41, 0x53,                                // push r11
        0x49, 0xba,                                // mov r10, record
    };
    emit_u64(code, reinterpret_cast<uintptr_t>(record));
    code.insert(code.end(), {0x41, 0xbb, 0x01, 0x00, 0x00, 0x00}); // r11 = 1
    code.insert(code.end(), {0x4d, 0x0f, 0xc1, 0x5a, 0x00});       // old count -> r11
    code.insert(code.end(), {0x4d, 0x85, 0xdb});                   // test r11,r11
    code.insert(code.end(), {0x0f, 0x85});                         // jne skip_first_sample (rel32)
    const size_t skip_offset = code.size();
    code.insert(code.end(), {0, 0, 0, 0});
    code.insert(code.end(), {0x49, 0x89, 0x4a, 0x08}); // [r10+08] = rcx
    code.insert(code.end(), {0x49, 0x89, 0x52, 0x10}); // [r10+10] = rdx
    code.insert(code.end(), {0x4d, 0x89, 0x42, 0x18}); // [r10+18] = r8
    code.insert(code.end(), {0x4d, 0x89, 0x4a, 0x20}); // [r10+20] = r9
    code.insert(code.end(), {0x49, 0x89, 0x5a, 0x60}); // [r10+60] = rbx
    code.insert(code.end(), {0x49, 0x89, 0x6a, 0x68}); // [r10+68] = rbp
    code.insert(code.end(), {0x49, 0x89, 0x72, 0x70}); // [r10+70] = rsi
    code.insert(code.end(), {0x49, 0x89, 0x7a, 0x78}); // [r10+78] = rdi
    code.insert(code.end(), {0x4d, 0x89, 0xa2, 0x80, 0x00, 0x00, 0x00}); // [r10+80] = r12
    code.insert(code.end(), {0x4d, 0x89, 0xaa, 0x88, 0x00, 0x00, 0x00}); // [r10+88] = r13
    code.insert(code.end(), {0x4d, 0x89, 0xb2, 0x90, 0x00, 0x00, 0x00}); // [r10+90] = r14
    code.insert(code.end(), {0x4d, 0x89, 0xba, 0x98, 0x00, 0x00, 0x00}); // [r10+98] = r15
    code.insert(code.end(), {0x4c, 0x8b, 0x1c, 0x24}); // r11 = original r11
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xa0, 0x00, 0x00, 0x00}); // [r10+a0] = r11
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x08}); // r11 = original r10
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xa8, 0x00, 0x00, 0x00}); // [r10+a8] = r11
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x18}); // r11 = return address
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xb0, 0x00, 0x00, 0x00}); // [r10+b0] = return address
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x40}); // r11=[rsp+40] (arg5)
    code.insert(code.end(), {0x4d, 0x89, 0x5a, 0x28}); // [r10+28] = arg5
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x48}); // r11=[rsp+48]
    code.insert(code.end(), {0x4d, 0x89, 0x5a, 0x30}); // [r10+30] = arg6
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x50}); // r11=[rsp+50]
    code.insert(code.end(), {0x4d, 0x89, 0x5a, 0x38}); // [r10+38] = arg7
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x58}); // r11=[rsp+58]
    code.insert(code.end(), {0x4d, 0x89, 0x5a, 0x40}); // [r10+40] = arg8
    const size_t skip_target = code.size();
    const int64_t skip_distance = static_cast<int64_t>(skip_target) -
                                  static_cast<int64_t>(skip_offset + sizeof(int32_t));
    if (skip_distance < std::numeric_limits<int32_t>::min() ||
        skip_distance > std::numeric_limits<int32_t>::max()) return nullptr;
    const uint32_t encoded_distance = static_cast<uint32_t>(static_cast<int32_t>(skip_distance));
    for (unsigned int i = 0; i < sizeof(int32_t); ++i)
        code[skip_offset + i] = static_cast<uint8_t>(encoded_distance >> (i * 8));
    code.insert(code.end(), {0x49, 0x89, 0x8a, 0xb8, 0x00, 0x00, 0x00}); // latest rcx
    code.insert(code.end(), {0x49, 0x89, 0x92, 0xc0, 0x00, 0x00, 0x00}); // latest rdx
    code.insert(code.end(), {0x4d, 0x89, 0x82, 0xc8, 0x00, 0x00, 0x00}); // latest r8
    code.insert(code.end(), {0x4d, 0x89, 0x8a, 0xd0, 0x00, 0x00, 0x00}); // latest r9
    code.insert(code.end(), {0x4c, 0x8b, 0x1c, 0x24}); // r11 = original r11
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0x38, 0x01, 0x00, 0x00}); // latest r11
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x08}); // r11 = original r10
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0x40, 0x01, 0x00, 0x00}); // latest r10
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x18}); // r11 = return address
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0x48, 0x01, 0x00, 0x00}); // latest return address
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x40}); // latest arg5
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xd8, 0x00, 0x00, 0x00});
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x48}); // latest arg6
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xe0, 0x00, 0x00, 0x00});
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x50}); // latest arg7
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xe8, 0x00, 0x00, 0x00});
    code.insert(code.end(), {0x4c, 0x8b, 0x5c, 0x24, 0x58}); // latest arg8
    code.insert(code.end(), {0x4d, 0x89, 0x9a, 0xf0, 0x00, 0x00, 0x00});
    code.insert(code.end(), {0x49, 0x89, 0x9a, 0xf8, 0x00, 0x00, 0x00}); // latest rbx
    code.insert(code.end(), {0x49, 0x89, 0xaa, 0x00, 0x01, 0x00, 0x00}); // latest rbp
    code.insert(code.end(), {0x49, 0x89, 0xb2, 0x08, 0x01, 0x00, 0x00}); // latest rsi
    code.insert(code.end(), {0x49, 0x89, 0xba, 0x10, 0x01, 0x00, 0x00}); // latest rdi
    code.insert(code.end(), {0x4d, 0x89, 0xa2, 0x18, 0x01, 0x00, 0x00}); // latest r12
    code.insert(code.end(), {0x4d, 0x89, 0xaa, 0x20, 0x01, 0x00, 0x00}); // latest r13
    code.insert(code.end(), {0x4d, 0x89, 0xb2, 0x28, 0x01, 0x00, 0x00}); // latest r14
    code.insert(code.end(), {0x4d, 0x89, 0xba, 0x30, 0x01, 0x00, 0x00}); // latest r15
    code.insert(code.end(), {0x41, 0x5b});             // pop r11
    code.insert(code.end(), {0x41, 0x5a});             // pop r10
    code.insert(code.end(), {0x58});                   // pop rax
    code.insert(code.end(), {0x48, 0xb8});         // mov rax, original
    emit_u64(code, reinterpret_cast<uintptr_t>(original));
    code.insert(code.end(), {0xff, 0xe0});         // jmp rax

    void *memory = VirtualAlloc(nullptr, code.size(), MEM_COMMIT | MEM_RESERVE,
                                PAGE_EXECUTE_READWRITE);
    if (!memory) return nullptr;
    std::memcpy(memory, code.data(), code.size());
    FlushInstructionCache(GetCurrentProcess(), memory, code.size());
    return memory;
}

void proxy_dark_table(const void **table_address, const void *uuid) {
    if (!table_address || !*table_address) return;
    const auto *original = static_cast<const uintptr_t *>(*table_address);
    size_t bytes = 0;
    std::memcpy(&bytes, original, sizeof(bytes));
    if (bytes == 0 || bytes > 0x10000 || bytes % sizeof(uintptr_t) != 0) {
        log([&] {
            std::ostringstream s;
            s << "dark_table_unproxied uuid=0x" << std::hex << reinterpret_cast<uintptr_t>(uuid)
              << " header=0x" << bytes;
            return s.str();
        });
        return;
    }

    std::lock_guard<std::mutex> lock(g_dark_mutex);
    for (const DarkTableProxy &existing : g_dark_tables) {
        if (existing.original == *table_address) {
            *table_address = existing.replacement;
            return;
        }
    }

    const size_t entries = bytes / sizeof(uintptr_t);
    auto *replacement = static_cast<uintptr_t *>(VirtualAlloc(
        nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!replacement) return;
    std::memcpy(replacement, original, bytes);
    DarkTableProxy proxy;
    proxy.original = const_cast<void *>(*table_address);
    proxy.replacement = replacement;
    proxy.bytes = bytes;
    proxy.thunks.resize(entries, nullptr);
    for (size_t i = 1; i < entries; ++i) {
        void *entry = reinterpret_cast<void *>(original[i]);
        DarkCallRecord *record_ptr = nullptr;
        if (g_dark_trace_view && g_dark_trace_record_cursor < kDarkTraceMaxRecords) {
            auto *records = reinterpret_cast<DarkCallRecord *>(
                static_cast<uint8_t *>(g_dark_trace_view) + sizeof(DarkTraceHeader));
            record_ptr = &records[g_dark_trace_record_cursor++];
            std::memset(record_ptr, 0, sizeof(*record_ptr));
        } else {
            auto record = std::make_unique<DarkCallRecord>();
            record_ptr = record.get();
            g_dark_owned_records.push_back(std::move(record));
        }
        record_ptr->table = replacement;
        record_ptr->index = i;
        record_ptr->original = entry;
        g_dark_records.push_back(record_ptr);
        void *thunk = make_dark_thunk(entry, record_ptr);
        if (thunk) {
            replacement[i] = reinterpret_cast<uintptr_t>(thunk);
            proxy.thunks[i] = thunk;
        }
    }
    *table_address = replacement;
    log([&] {
        std::ostringstream s;
        s << "dark_table_proxy uuid=0x" << std::hex << reinterpret_cast<uintptr_t>(uuid)
          << " original=0x" << reinterpret_cast<uintptr_t>(original)
          << " replacement=0x" << reinterpret_cast<uintptr_t>(replacement)
          << " bytes=" << std::dec << bytes << " entries=" << entries;
        return s.str();
    });
    g_dark_tables.push_back(std::move(proxy));
    if (g_dark_trace_header) g_dark_trace_header->record_count = g_dark_trace_record_cursor;
}

void dump_dark_records() {
    for (const DarkCallRecord *entry : g_dark_records) {
        if (entry->count == 0) continue;
        log([&] {
            std::ostringstream s;
            s << "dark_table_summary table=0x" << std::hex
              << reinterpret_cast<uintptr_t>(entry->table) << " index=" << std::dec
              << entry->index << " count=" << entry->count << " original=0x" << std::hex
              << reinterpret_cast<uintptr_t>(entry->original) << " regs=";
            for (uintptr_t value : entry->register_args) s << "0x" << std::hex << value << ',';
            s << " stack=";
            for (uintptr_t value : entry->stack_args) s << "0x" << std::hex << value << ',';
            s << " nonvolatile=";
            for (uintptr_t value : entry->nonvolatile_args) s << "0x" << std::hex << value << ',';
            s << " extra=0x" << entry->extra_args[0] << ",0x" << entry->extra_args[1]
              << " return=0x" << entry->return_address << " caller_module="
              << address_module(entry->return_address) << " caller_base=0x" << std::hex
              << address_module_base(entry->return_address);
            return s.str();
        });
        char scan[4]{};
        char scan_all[4]{};
        const bool scan_enabled = GetEnvironmentVariableA("DLSS5_DARK_SCAN", scan, sizeof(scan)) > 0;
        const bool scan_every_slot = GetEnvironmentVariableA("DLSS5_DARK_SCAN_ALL", scan_all, sizeof(scan_all)) > 0;
        if (scan_enabled && (scan_every_slot || entry->index == 12 || entry->index == 50)) {
            for (uintptr_t value : entry->register_args) {
                dump_dark_memory("dark_table_memory", value);
                dump_dark_binary("dark_table_memory", value);
            }
            for (uintptr_t value : entry->stack_args) {
                dump_dark_memory("dark_table_stack_memory", value);
                dump_dark_binary("dark_table_stack_memory", value);
            }
        }
    }
}

CUresult __stdcall hook_cu_module_load_data(CUmodule *module, const void *image) {
    dump_cuda_image("cuModuleLoadData", image);
    return g_cu_module_load_data ? g_cu_module_load_data(module, image) : 1;
}

CUresult __stdcall hook_cu_module_load_data_ex(CUmodule *module, const void *image,
                                               unsigned int count, void *options, void *values) {
    dump_cuda_image("cuModuleLoadDataEx", image);
    return g_cu_module_load_data_ex ? g_cu_module_load_data_ex(module, image, count, options, values) : 1;
}

CUresult __stdcall hook_cu_module_get_function(CUfunction *function, CUmodule module, const char *name) {
    log([&] {
        std::ostringstream s;
        s << "cuModuleGetFunction module=0x" << std::hex << reinterpret_cast<uintptr_t>(module)
          << " name=" << (name ? name : "<null>");
        return s.str();
    });
    return g_cu_module_get_function ? g_cu_module_get_function(function, module, name) : 1;
}

CUresult __stdcall hook_cu_get_export_table(const void **table, const void *uuid) {
    const CUresult result = g_cu_get_export_table ? g_cu_get_export_table(table, uuid) : 1;
    if (result == 0) proxy_dark_table(table, uuid);
    log([&] {
        std::ostringstream s;
        s << "cuGetExportTable uuid=0x" << std::hex << reinterpret_cast<uintptr_t>(uuid)
          << " result=" << std::dec << result << " table=0x";
        if (table) s << std::hex << reinterpret_cast<uintptr_t>(*table);
        else s << 0;
        if (uuid) s << " uuid_bytes=" << hex_bytes(uuid, 16);
        if (table && *table) {
            s << " entries=";
            const auto *entries = static_cast<const uintptr_t *>(*table);
            for (unsigned int i = 0; i < 64; ++i) s << "0x" << std::hex << entries[i] << ',';
        }
        return s.str();
    });
    return result;
}

CUresult __stdcall hook_cu_launch_kernel(CUfunction function, unsigned int gx, unsigned int gy,
                                         unsigned int gz, unsigned int bx, unsigned int by,
                                         unsigned int bz, unsigned int shared, CUstream stream,
                                         void **args, void **extra) {
    log([&] {
        std::ostringstream s;
        s << "cuLaunchKernel function=0x" << std::hex << reinterpret_cast<uintptr_t>(function)
          << " grid=" << std::dec << gx << ',' << gy << ',' << gz
          << " block=" << bx << ',' << by << ',' << bz << " shared=" << shared
          << " stream=0x" << std::hex << reinterpret_cast<uintptr_t>(stream)
          << " args=" << reinterpret_cast<uintptr_t>(args)
          << " extra=" << reinterpret_cast<uintptr_t>(extra);
        if (args) {
            s << " arg_values=";
            for (unsigned int i = 0; i < 32; ++i) {
                if (!args[i]) { s << "null,"; break; }
                s << "0x" << std::hex << *reinterpret_cast<uint64_t *>(args[i]) << ',';
            }
        }
        return s.str();
    });
    return g_cu_launch_kernel ? g_cu_launch_kernel(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra) : 1;
}

void hook_cuda_proc(const char *name, FARPROC address) {
    if (!name || !address) return;
    void *replacement = nullptr;
    void **original = nullptr;
    if (std::strcmp(name, "cuModuleLoadData") == 0) {
        replacement = reinterpret_cast<void *>(&hook_cu_module_load_data);
        original = reinterpret_cast<void **>(&g_cu_module_load_data);
    } else if (std::strcmp(name, "cuModuleLoadDataEx") == 0) {
        replacement = reinterpret_cast<void *>(&hook_cu_module_load_data_ex);
        original = reinterpret_cast<void **>(&g_cu_module_load_data_ex);
    } else if (std::strcmp(name, "cuModuleGetFunction") == 0) {
        replacement = reinterpret_cast<void *>(&hook_cu_module_get_function);
        original = reinterpret_cast<void **>(&g_cu_module_get_function);
    } else if (std::strcmp(name, "cuLaunchKernel") == 0) {
        replacement = reinterpret_cast<void *>(&hook_cu_launch_kernel);
        original = reinterpret_cast<void **>(&g_cu_launch_kernel);
    } else if (std::strcmp(name, "cuGetExportTable") == 0) {
        replacement = reinterpret_cast<void *>(&hook_cu_get_export_table);
        original = reinterpret_cast<void **>(&g_cu_get_export_table);
    }
    log([&] {
        std::ostringstream s;
        s << "cuda_getproc name=" << name << " address=0x" << std::hex
          << reinterpret_cast<uintptr_t>(address);
        return s.str();
    });
    if (replacement && original) {
        const MH_STATUS status = MH_CreateHook(reinterpret_cast<LPVOID>(address), replacement, original);
        if (status == MH_OK || status == MH_ERROR_ALREADY_CREATED) {
            MH_EnableHook(reinterpret_cast<LPVOID>(address));
            log([&] { return std::string("cuda_hook_installed ") + name; });
        } else {
            log([&] {
                std::ostringstream s; s << "cuda_hook_failed " << name << " status=" << int(status); return s.str();
            });
        }
    }
}

void hook_cuda_module(HMODULE module) {
    if (!module || !g_get_proc_address) return;
    static const char *const names[] = {
        "cuGetExportTable", "cuGetProcAddress", "cuModuleLoadData", "cuModuleLoadDataEx",
        "cuModuleGetFunction", "cuModuleUnload", "cuLaunch", "cuLaunchKernel",
        "cuLaunchKernel_ptsz", "cuLaunchKernelEx", "cuLaunchKernelEx_ptsz",
        "cuGraphLaunch", "cuStreamSynchronize", "cuCtxSynchronize",
    };
    for (const char *name : names) {
        const FARPROC address = g_get_proc_address(module, name);
        if (address) hook_cuda_proc(name, address);
    }
    log([&] { return std::string("cuda_exports_hooked module=") + narrow(module_name(module).c_str()); });
}

FARPROC WINAPI hook_get_proc_address(HMODULE module, LPCSTR name) {
    const FARPROC address = g_get_proc_address ? g_get_proc_address(module, name) : nullptr;
    const std::wstring loaded = module_name(module);
    if ((loaded.find(L"nvcuda") != std::wstring::npos || loaded.find(L"nvapi") != std::wstring::npos) && name) {
        hook_cuda_proc(name, address);
    }
    return address;
}

HMODULE WINAPI hook_load_library_ex_w(LPCWSTR name, HANDLE file, DWORD flags) {
    const HMODULE module = g_load_library_ex_w ? g_load_library_ex_w(name, file, flags) : nullptr;
    if (module) log([&] {
        std::ostringstream s; s << "load_library_ex " << narrow(name) << " module=0x"
                                << std::hex << reinterpret_cast<uintptr_t>(module); return s.str();
    });
    if (module && module_name(module).find(L"nvcuda") != std::wstring::npos) hook_cuda_module(module);
    return module;
}

HMODULE WINAPI hook_load_library_w(LPCWSTR name) {
    const HMODULE module = g_load_library_w ? g_load_library_w(name) : nullptr;
    if (module) log([&] {
        std::ostringstream s; s << "load_library " << narrow(name) << " module=0x"
                                << std::hex << reinterpret_cast<uintptr_t>(module); return s.str();
    });
    if (module && module_name(module).find(L"nvcuda") != std::wstring::npos) hook_cuda_module(module);
    return module;
}

void install_driver_hooks() {
    std::call_once(g_hooks_once, [] {
        const MH_STATUS init = MH_Initialize();
        log([&] { std::ostringstream s; s << "minhook_initialize status=" << int(init); return s.str(); });
        if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) return;
        MH_CreateHookApi(L"kernel32.dll", "GetProcAddress", reinterpret_cast<LPVOID>(&hook_get_proc_address), reinterpret_cast<LPVOID *>(&g_get_proc_address));
        MH_CreateHookApi(L"kernel32.dll", "LoadLibraryExW", reinterpret_cast<LPVOID>(&hook_load_library_ex_w), reinterpret_cast<LPVOID *>(&g_load_library_ex_w));
        MH_CreateHookApi(L"kernel32.dll", "LoadLibraryW", reinterpret_cast<LPVOID>(&hook_load_library_w), reinterpret_cast<LPVOID *>(&g_load_library_w));
        MH_EnableHook(MH_ALL_HOOKS);
        log([] { return std::string("driver_hooks_enabled"); });
    });
}

void on_init_device(device *dev) {
    init_dark_trace();
    install_driver_hooks();
    install_d3d12_device_hooks(dev);
    log([&] {
        std::ostringstream s;
        char description[256]{};
        uint32_t vendor = 0, device_id = 0;
        dev->get_property(device_properties::description, description);
        dev->get_property(device_properties::vendor_id, &vendor);
        dev->get_property(device_properties::device_id, &device_id);
        s << "init_device api=0x" << std::hex << uint32_t(dev->get_api())
          << " vendor=0x" << vendor << " device=0x" << device_id
          << " description=" << description << std::dec;
        return s.str();
    });
}

void on_init_command_list(command_list *cmd_list) {
    install_d3d12_command_list_hooks(cmd_list);
    log([&] {
        std::ostringstream s;
        s << "init_command_list native=0x" << std::hex
          << (cmd_list ? cmd_list->get_native() : 0);
        return s.str();
    });
}

void on_init_resource(device *dev, const resource_desc &desc,
                      const subresource_data *, resource_usage initial_state,
                      resource resource_handle) {
    log([&] {
        std::ostringstream s;
        s << "init_resource handle=0x" << std::hex << resource_handle.handle
          << " type=" << uint32_t(desc.type) << " usage=0x" << uint32_t(desc.usage)
          << " state=0x" << uint32_t(initial_state) << " flags=0x" << uint32_t(desc.flags)
          << std::dec;
        if (dev->get_api() == device_api::d3d12 && resource_handle.handle != 0) {
            auto *resource = reinterpret_cast<ID3D12Resource *>(resource_handle.handle);
            s << " native=0x" << std::hex << resource_handle.handle
              << " gpu_va=0x" << resource->GetGPUVirtualAddress() << std::dec;
        }
        if (desc.type == resource_type::buffer) {
            s << " bytes=" << desc.buffer.size;
        } else {
            s << " width=" << desc.texture.width << " height=" << desc.texture.height
              << " layers=" << desc.texture.depth_or_layers << " levels=" << desc.texture.levels
              << " format=" << uint32_t(desc.texture.format);
        }
        return s.str();
    });
}

void on_init_resource_view(device *dev, resource resource_handle, resource_usage usage,
                           const resource_view_desc &desc, resource_view view) {
    log([&] {
        std::ostringstream s;
        s << "init_view handle=0x" << std::hex << view.handle
          << " resource=0x" << resource_handle.handle << " usage=0x" << uint32_t(usage)
          << " type=" << uint32_t(desc.type) << " format=" << uint32_t(desc.format)
          << std::dec;
        if (desc.type == resource_view_type::buffer) {
            s << " offset=" << desc.buffer.offset << " size=" << desc.buffer.size;
        } else {
            s << " first_level=" << desc.texture.first_level << " levels=" << desc.texture.levels
              << " first_layer=" << desc.texture.first_layer << " layers=" << desc.texture.layers;
        }
        return s.str();
    });
}

void on_init_pipeline_layout(device *, uint32_t count, const pipeline_layout_param *params,
                             pipeline_layout layout) {
    log([&] {
        std::ostringstream s;
        s << "init_layout handle=0x" << std::hex << layout.handle << std::dec
          << " params=" << count;
        for (uint32_t i = 0; i < count; ++i) {
            s << " p" << i << "=" << uint32_t(params[i].type);
            if (params[i].type == pipeline_layout_param_type::push_constants) {
                s << ":cbinding=" << params[i].push_constants.binding
                  << ":count=" << params[i].push_constants.count;
            }
        }
        return s.str();
    });
}

void on_push_constants(command_list *, shader_stage stages, pipeline_layout layout,
                       uint32_t param, uint32_t first, uint32_t count, const void *values) {
    log([&] {
        std::ostringstream s;
        s << "push_constants stages=0x" << std::hex << uint32_t(stages)
          << " layout=0x" << layout.handle << " param=" << std::dec << param
          << " first=" << first << " count=" << count
          << " bytes=" << hex_bytes(values, size_t(count) * 4);
        return s.str();
    });
}

void on_push_descriptors(command_list *, shader_stage stages, pipeline_layout layout,
                         uint32_t param, const descriptor_table_update &update) {
    log([&] {
        std::ostringstream s;
        s << "push_descriptors stages=0x" << std::hex << uint32_t(stages)
          << " layout=0x" << layout.handle << " param=" << std::dec << param
          << " table=0x" << std::hex << update.table.handle << std::dec
          << " binding=" << update.binding << " array=" << update.array_offset
          << " count=" << update.count << " type=" << uint32_t(update.type);
        if (update.descriptors != nullptr && update.count != 0) {
            if (update.type == descriptor_type::shader_resource_view ||
                update.type == descriptor_type::unordered_access_view) {
                const auto *views = static_cast<const resource_view *>(update.descriptors);
                s << " views=";
                for (uint32_t i = 0; i < update.count; ++i)
                    s << "0x" << std::hex << views[i].handle << ',';
            } else if (update.type == descriptor_type::constant_buffer ||
                       update.type == descriptor_type::shader_storage_buffer) {
                const auto *ranges = static_cast<const buffer_range *>(update.descriptors);
                s << " buffers=";
                for (uint32_t i = 0; i < update.count; ++i)
                    s << "0x" << std::hex << ranges[i].buffer.handle << '+'
                      << std::dec << ranges[i].offset << '/' << ranges[i].size << ',';
            }
        }
        return s.str();
    });
}

void on_bind_descriptor_tables(command_list *, shader_stage stages, pipeline_layout layout,
                              uint32_t first, uint32_t count, const descriptor_table *tables,
                              uint32_t dynamic_count, const uint32_t *dynamic_offsets) {
    log([&] {
        std::ostringstream s;
        s << "bind_tables stages=0x" << std::hex << uint32_t(stages)
          << " layout=0x" << layout.handle << std::dec << " first=" << first
          << " count=" << count << " dynamic_count=" << dynamic_count << " tables=";
        for (uint32_t i = 0; i < count; ++i) s << "0x" << std::hex << tables[i].handle << ',';
        if (dynamic_offsets != nullptr) {
            s << " offsets=";
            for (uint32_t i = 0; i < dynamic_count; ++i) s << std::dec << dynamic_offsets[i] << ',';
        }
        return s.str();
    });
}

void on_bind_pipeline(command_list *, pipeline_stage stage, pipeline pipeline_handle) {
    log([&] {
        std::ostringstream s;
        s << "bind_pipeline stage=0x" << std::hex << uint32_t(stage)
          << " pipeline=0x" << pipeline_handle.handle;
        return s.str();
    });
}

bool on_dispatch(command_list *, uint32_t x, uint32_t y, uint32_t z) {
    log([&] {
        std::ostringstream s;
        s << "dispatch " << x << ' ' << y << ' ' << z;
        return s.str();
    });
    return false;
}

void on_reshade_begin_effects(effect_runtime *, command_list *, resource_view, resource_view) {
    log([] { return std::string("reshade_begin_effects"); });
}

void on_reshade_finish_effects(effect_runtime *, command_list *, resource_view, resource_view) {
    log([] { return std::string("reshade_finish_effects"); });
}

void on_execute_command_list(command_queue *, command_list *) {
    log([] { return std::string("execute_command_list"); });
}

} // namespace

extern "C" __declspec(dllexport) const char *NAME = "DLSS5 Capture";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "DLSS5 graphics ABI capture";

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        if (!reshade::register_addon(hModule)) return FALSE;
        log([&] {
            std::ostringstream s; s << "process_attach pid=" << GetCurrentProcessId(); return s.str();
        });
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::init_command_list>(on_init_command_list);
        reshade::register_event<reshade::addon_event::init_resource>(on_init_resource);
        reshade::register_event<reshade::addon_event::init_resource_view>(on_init_resource_view);
        reshade::register_event<reshade::addon_event::init_pipeline_layout>(on_init_pipeline_layout);
        reshade::register_event<reshade::addon_event::push_constants>(on_push_constants);
        reshade::register_event<reshade::addon_event::push_descriptors>(on_push_descriptors);
        reshade::register_event<reshade::addon_event::bind_descriptor_tables>(on_bind_descriptor_tables);
        reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::execute_command_list>(on_execute_command_list);
        reshade::register_event<reshade::addon_event::reshade_begin_effects>(on_reshade_begin_effects);
        reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_reshade_finish_effects);
    } else if (reason == DLL_PROCESS_DETACH) {
        dump_dark_records();
        close_dark_trace();
        reshade::unregister_addon(hModule);
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_log.is_open()) g_log.close();
    }
    return TRUE;
}
