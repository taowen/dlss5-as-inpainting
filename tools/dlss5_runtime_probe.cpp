// Minimal native D3D12 probe for the pinned DLSS-NR Feature 18 DLL.
//
// This intentionally uses the driver's public nvngx.dll for parameter/feature
// dispatch.  The local nvngx_dlssnr.dll is supplied through FeatureCommonInfo,
// so the model is reached by the same NGX path a host application uses.  No
// PyTorch or CUDA runtime is involved in this probe.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kSuccess = 0x1;
constexpr int kFeature18 = 18;
constexpr int kVersion15 = 0x0000015;

// These are the stable parts of the public NGX definitions needed by this
// probe.  The full SDK header is intentionally not vendored into the project.
struct PathListInfo {
    const wchar_t* const* Path;
    unsigned int Length;
};

struct LoggingInfo {
    void* LoggingCallback;
    int MinimumLoggingLevel;
    bool DisableOtherLoggingSinks;
    std::uint8_t Padding[3];
};

struct FeatureCommonInfo {
    PathListInfo PathListInfo;
    void* InternalData;
    LoggingInfo LoggingInfo;
};

struct Parameter;
struct Handle;

using InitFn = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*,
                             int, const FeatureCommonInfo*);
using GetCapabilityParametersFn = int(__cdecl*)(Parameter**);
using DestroyParametersFn = int(__cdecl*)(Parameter*);
using CreateFeatureFn = int(__cdecl*)(ID3D12GraphicsCommandList*, int, Parameter*, Handle**);
using EvaluateFeatureFn = int(__cdecl*)(ID3D12GraphicsCommandList*, const Handle*, const Parameter*, void*);
using ReleaseFeatureFn = int(__cdecl*)(Handle*);
using Shutdown1Fn = int(__cdecl*)(ID3D12Device*);

using ForwarderCreateFn = void*(__cdecl*)(const wchar_t*, const wchar_t*, ID3D12Device*,
                                          ID3D12GraphicsCommandList*, void*, unsigned int, unsigned int,
                                          int, float, int, float, float, float, int, int);
using ForwarderEvaluateFn = int(__cdecl*)(ID3D12GraphicsCommandList*, void*, void*, ID3D12Resource*,
                                          ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, unsigned int,
                                          unsigned int, unsigned int, unsigned int, int, int, float, int,
                                          float, float, float, int, float, float);
using ForwarderReleaseFn = void(__cdecl*)(void*);

using SetULLFn = void(__thiscall*)(void*, const char*, unsigned long long);
using SetFloatFn = void(__thiscall*)(void*, const char*, float);
using SetUIntFn = void(__thiscall*)(void*, const char*, unsigned int);
using SetIntFn = void(__thiscall*)(void*, const char*, int);
using GetFloatFn = int(__thiscall*)(void*, const char*, float*);

struct NgxApi {
    HMODULE module = nullptr;
    InitFn init = nullptr;
    GetCapabilityParametersFn getCapabilityParameters = nullptr;
    DestroyParametersFn destroyParameters = nullptr;
    CreateFeatureFn createFeature = nullptr;
    EvaluateFeatureFn evaluateFeature = nullptr;
    ReleaseFeatureFn releaseFeature = nullptr;
    Shutdown1Fn shutdown1 = nullptr;
    HMODULE forwarderModule = nullptr;
    ForwarderCreateFn forwarderCreate = nullptr;
    ForwarderEvaluateFn forwarderEvaluate = nullptr;
    ForwarderReleaseFn forwarderRelease = nullptr;
    bool useForwarder = false;
};

struct Texture {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct Readback {
    ComPtr<ID3D12Resource> resource;
    UINT64 rowPitch = 0;
    UINT64 totalBytes = 0;
};

struct ProbeConfig {
    unsigned int width = 256;
    unsigned int height = 256;
    std::filesystem::path root;
    std::filesystem::path outputDir;
    std::filesystem::path forwarder;
    std::filesystem::path driverNvngx;
    bool useForwarder = false;
};

struct ProbeContext {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceValue = 0;
};

struct Resources {
    Texture color;
    Texture depth;
    Texture motion;
    Texture controlMask;
    Texture output;
};

void fail(const std::string& message, HRESULT hr = S_OK) {
    if (FAILED(hr)) {
        std::cerr << message << " HRESULT=0x" << std::hex << static_cast<unsigned long>(hr)
                  << std::dec << "\n";
    } else {
        std::cerr << message << "\n";
    }
    std::exit(2);
}

template <typename T>
void checkHr(HRESULT hr, const char* message) {
    if (FAILED(hr)) fail(message, hr);
}

void checkNgx(int result, const char* message) {
    if (result != kSuccess) {
        std::cerr << message << " result=0x" << std::hex << static_cast<unsigned int>(result)
                  << std::dec << "\n";
        std::exit(3);
    }
}

template <typename T>
T proc(HMODULE module, const char* name) {
    auto value = GetProcAddress(module, name);
    if (!value) fail(std::string("missing export: ") + name);
    return reinterpret_cast<T>(value);
}

std::uint16_t floatToHalf(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        return static_cast<std::uint16_t>(sign | ((mantissa + (1u << (shift - 1))) >> shift));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                       ((mantissa + 0x1000u) >> 13));
}

void resetCommandList(ProbeContext& ctx) {
    checkHr<void>(ctx.allocator->Reset(), "command allocator reset failed");
    checkHr<void>(ctx.commandList->Reset(ctx.allocator.Get(), nullptr), "command list reset failed");
}

void submitAndWait(ProbeContext& ctx) {
    checkHr<void>(ctx.commandList->Close(), "command list close failed");
    ID3D12CommandList* lists[] = {ctx.commandList.Get()};
    ctx.queue->ExecuteCommandLists(1, lists);
    ++ctx.fenceValue;
    checkHr<void>(ctx.queue->Signal(ctx.fence.Get(), ctx.fenceValue), "queue signal failed");
    if (ctx.fence->GetCompletedValue() < ctx.fenceValue) {
        checkHr<void>(ctx.fence->SetEventOnCompletion(ctx.fenceValue, ctx.fenceEvent),
                      "fence event failed");
        WaitForSingleObject(ctx.fenceEvent, INFINITE);
    }
}

void transition(ProbeContext& ctx, Texture& texture, D3D12_RESOURCE_STATES next) {
    if (texture.state == next) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.resource.Get();
    barrier.Transition.StateBefore = texture.state;
    barrier.Transition.StateAfter = next;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx.commandList->ResourceBarrier(1, &barrier);
    texture.state = next;
}

Texture createTexture(ProbeContext& ctx, unsigned int width, unsigned int height,
                      DXGI_FORMAT format, D3D12_RESOURCE_STATES initial) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    Texture texture;
    checkHr<void>(ctx.device->CreateCommittedResource(
                      &heap, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr,
                      IID_PPV_ARGS(&texture.resource)),
                  "texture allocation failed");
    texture.state = initial;
    return texture;
}

Readback createReadback(ProbeContext& ctx, unsigned int width, unsigned int height,
                        DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    UINT64 rowSize = 0;
    UINT64 total = 0;
    ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, &rowSize, &total);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.Format = DXGI_FORMAT_UNKNOWN;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    Readback result;
    checkHr<void>(ctx.device->CreateCommittedResource(
                      &heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                      nullptr, IID_PPV_ARGS(&result.resource)),
                  "readback allocation failed");
    result.rowPitch = rowSize;
    result.totalBytes = total;
    return result;
}

void uploadTexture(ProbeContext& ctx, Texture& texture, unsigned int width, unsigned int height,
                   DXGI_FORMAT format, const void* data, size_t rowBytes) {
    D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
    UINT64 uploadSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowSize = 0;
    ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowSize, &uploadSize);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = uploadSize;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.Format = DXGI_FORMAT_UNKNOWN;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upload;
    checkHr<void>(ctx.device->CreateCommittedResource(
                      &heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                      nullptr, IID_PPV_ARGS(&upload)),
                  "upload allocation failed");

    std::uint8_t* mapped = nullptr;
    D3D12_RANGE noRead{0, 0};
    checkHr<void>(upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped)), "upload map failed");
    const auto* source = static_cast<const std::uint8_t*>(data);
    for (UINT y = 0; y < height; ++y) {
        std::memcpy(mapped + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch,
                    source + static_cast<size_t>(y) * rowBytes, rowBytes);
    }
    upload->Unmap(0, nullptr);

    transition(ctx, texture, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = texture.resource.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION sourceLocation{};
    sourceLocation.pResource = upload.Get();
    sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceLocation.PlacedFootprint = footprint;
    ctx.commandList->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
    transition(ctx, texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void copyToReadback(ProbeContext& ctx, Texture& texture, Readback& readback) {
    transition(ctx, texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = texture.resource.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.resource.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Offset = 0;
    destination.PlacedFootprint.Footprint.Format = texture.resource->GetDesc().Format;
    destination.PlacedFootprint.Footprint.Width = texture.resource->GetDesc().Width;
    destination.PlacedFootprint.Footprint.Height = texture.resource->GetDesc().Height;
    destination.PlacedFootprint.Footprint.Depth = 1;
    destination.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(readback.rowPitch);
    ctx.commandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
}

std::vector<std::uint8_t> readback(ProbeContext& ctx, Readback& resource) {
    std::vector<std::uint8_t> bytes(static_cast<size_t>(resource.totalBytes));
    D3D12_RANGE range{0, resource.totalBytes};
    void* mapped = nullptr;
    checkHr<void>(resource.resource->Map(0, &range, &mapped), "readback map failed");
    std::memcpy(bytes.data(), mapped, bytes.size());
    resource.resource->Unmap(0, nullptr);
    return bytes;
}

void setUInt(Parameter* params, const char* name, unsigned int value) {
    void** vtable = *reinterpret_cast<void***>(params);
    reinterpret_cast<SetUIntFn>(vtable[3])(params, name, value);
}

void setInt(Parameter* params, const char* name, int value) {
    void** vtable = *reinterpret_cast<void***>(params);
    reinterpret_cast<SetIntFn>(vtable[4])(params, name, value);
}

void setFloat(Parameter* params, const char* name, float value, int floatSlot) {
    void** vtable = *reinterpret_cast<void***>(params);
    reinterpret_cast<SetFloatFn>(vtable[floatSlot])(params, name, value);
}

void setResource(Parameter* params, const char* name, ID3D12Resource* resource) {
    void** vtable = *reinterpret_cast<void***>(params);
    reinterpret_cast<SetULLFn>(vtable[0])(params, name, reinterpret_cast<unsigned long long>(resource));
}

bool tryFloatRoundTrip(Parameter* params, void** vtable, int slot) {
    // The parameter ABI is a C++ vtable. Calling a non-float setter with a
    // float signature is undefined on x64, so keep each candidate behind SEH.
    // This also makes the probe useful across driver revisions whose vtable
    // layout is not identical to the public header.
    __try {
        reinterpret_cast<SetFloatFn>(vtable[slot])(params, "DLSSNR.Probe", 0.3125f);
        float roundTrip = 0.0f;
        auto getter = reinterpret_cast<GetFloatFn>(vtable[slot + 8]);
        return getter(params, "DLSSNR.Probe", &roundTrip) == kSuccess && roundTrip == 0.3125f;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

int discoverFloatSlot(Parameter* params) {
    void** vtable = *reinterpret_cast<void***>(params);
    // Slot 1 is the public NGX ABI's float setter. Test it first, then probe
    // the remaining setter slots only when a driver revision needs it.
    const int candidates[] = {1, 2, 3, 4, 5, 6, 7, 0};
    for (int slot : candidates) {
        if (tryFloatRoundTrip(params, vtable, slot)) {
            std::cout << "float parameter vtable slot=" << slot << "\n";
            return slot;
        }
    }
    std::cout << "float parameter slot probe failed; using slot 1\n";
    return 1;
}

ProbeContext createD3D12() {
    ProbeContext ctx;
    checkHr<void>(CreateDXGIFactory2(0, IID_PPV_ARGS(&ctx.factory)), "DXGI factory creation failed");
    for (UINT index = 0; ; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (ctx.factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                     IID_PPV_ARGS(&adapter)) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&ctx.device)) == S_OK) {
            ctx.adapter = adapter;
            std::wcout << L"adapter=" << desc.Description << L"\n";
            break;
        }
    }
    if (!ctx.device) fail("no D3D12 adapter could create a device");

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    checkHr<void>(ctx.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&ctx.queue)),
                  "command queue creation failed");
    checkHr<void>(ctx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                       IID_PPV_ARGS(&ctx.allocator)),
                  "command allocator creation failed");
    checkHr<void>(ctx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 ctx.allocator.Get(), nullptr,
                                                 IID_PPV_ARGS(&ctx.commandList)),
                  "command list creation failed");
    checkHr<void>(ctx.commandList->Close(), "initial command list close failed");
    checkHr<void>(ctx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx.fence)),
                  "fence creation failed");
    ctx.fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ctx.fenceEvent) fail("fence event creation failed");
    return ctx;
}

NgxApi loadNgx(const std::filesystem::path& path) {
    NgxApi api;
    api.module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!api.module) fail("could not load driver nvngx.dll");
    api.init = proc<InitFn>(api.module, "NVSDK_NGX_D3D12_Init_Ext");
    api.getCapabilityParameters = proc<GetCapabilityParametersFn>(
        api.module, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    api.destroyParameters = proc<DestroyParametersFn>(api.module, "NVSDK_NGX_D3D12_DestroyParameters");
    api.createFeature = proc<CreateFeatureFn>(api.module, "NVSDK_NGX_D3D12_CreateFeature");
    api.evaluateFeature = proc<EvaluateFeatureFn>(api.module, "NVSDK_NGX_D3D12_EvaluateFeature");
    api.releaseFeature = proc<ReleaseFeatureFn>(api.module, "NVSDK_NGX_D3D12_ReleaseFeature");
    api.shutdown1 = proc<Shutdown1Fn>(api.module, "NVSDK_NGX_D3D12_Shutdown1");
    return api;
}

void loadForwarder(NgxApi& api, const std::filesystem::path& path) {
    api.forwarderModule = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!api.forwarderModule) fail("could not load nvngx.dll_dlssnr.dll forwarder");
    api.forwarderCreate = proc<ForwarderCreateFn>(api.forwarderModule, "dlssnr_call_create");
    api.forwarderEvaluate = proc<ForwarderEvaluateFn>(api.forwarderModule, "dlssnr_call_evaluate");
    api.forwarderRelease = proc<ForwarderReleaseFn>(api.forwarderModule, "dlssnr_call_release");
    api.useForwarder = true;
}

void setControlMaskParams(Parameter* params, Resources& resources, unsigned int width,
                          unsigned int height, bool withControlMask) {
    setResource(params, "DLSSNR.ControlMask",
                withControlMask ? resources.controlMask.resource.Get() : nullptr);
    setUInt(params, "DLSSNR.ControlMaskSubrectBaseX", 0);
    setUInt(params, "DLSSNR.ControlMaskSubrectBaseY", 0);
    setUInt(params, "DLSSNR.ControlMaskSubrectWidth", width);
    setUInt(params, "DLSSNR.ControlMaskSubrectHeight", height);
}

void setEvaluateParams(Parameter* params, Resources& resources, unsigned int width,
                       unsigned int height, unsigned int guideWidth, unsigned int guideHeight,
                       int reset, int floatSlot, bool withControlMask) {
    setResource(params, "DLSSNR.Color", resources.color.resource.Get());
    setResource(params, "DLSSNR.Depth", resources.depth.resource.Get());
    setResource(params, "DLSSNR.MVec", resources.motion.resource.Get());
    setResource(params, "DLSSNR.Output", resources.output.resource.Get());
    if (withControlMask) {
        setResource(params, "DLSSNR.ControlMask", resources.controlMask.resource.Get());
    } else {
        setResource(params, "DLSSNR.ControlMask", nullptr);
    }
    setUInt(params, "DLSSNR.Enabled", 1);
    setUInt(params, "DLSSNR.Width", width);
    setUInt(params, "DLSSNR.Height", height);
    setUInt(params, "DLSSNR.DepthInverted", 0);
    setUInt(params, "DLSSNR.Reset", static_cast<unsigned int>(reset));
    setUInt(params, "DLSSNR.ColorSubrectBaseX", 0);
    setUInt(params, "DLSSNR.ColorSubrectBaseY", 0);
    setUInt(params, "DLSSNR.ColorSubrectWidth", width);
    setUInt(params, "DLSSNR.ColorSubrectHeight", height);
    setUInt(params, "DLSSNR.OutputSubrectBaseX", 0);
    setUInt(params, "DLSSNR.OutputSubrectBaseY", 0);
    setUInt(params, "DLSSNR.OutputSubrectWidth", width);
    setUInt(params, "DLSSNR.OutputSubrectHeight", height);
    setUInt(params, "DLSSNR.DepthSubrectBaseX", 0);
    setUInt(params, "DLSSNR.DepthSubrectBaseY", 0);
    setUInt(params, "DLSSNR.DepthSubrectWidth", guideWidth);
    setUInt(params, "DLSSNR.DepthSubrectHeight", guideHeight);
    setUInt(params, "DLSSNR.MVecSubrectBaseX", 0);
    setUInt(params, "DLSSNR.MVecSubrectBaseY", 0);
    setUInt(params, "DLSSNR.MVecSubrectWidth", guideWidth);
    setUInt(params, "DLSSNR.MVecSubrectHeight", guideHeight);
    setFloat(params, "DLSSNR.MVecScaleX", 1.0f, floatSlot);
    setFloat(params, "DLSSNR.MVecScaleY", 1.0f, floatSlot);
    setFloat(params, "DLSSNR.Intensity", 1.0f, floatSlot);
    setUInt(params, "DLSSNR.Style", 0);
    setFloat(params, "DLSSNR.LocalStructureStrength", 1.0f, floatSlot);
    setFloat(params, "DLSSNR.LocalToneStrength", 1.0f, floatSlot);
    setFloat(params, "DLSSNR.SkinStructureStrength", 1.0f, floatSlot);
    setUInt(params, "DLSSNR.UseAutoMask", 0);
    setUInt(params, "DLSSNR.ControlMaskSubrectBaseX", 0);
    setUInt(params, "DLSSNR.ControlMaskSubrectBaseY", 0);
    setUInt(params, "DLSSNR.ControlMaskSubrectWidth", width);
    setUInt(params, "DLSSNR.ControlMaskSubrectHeight", height);
}

std::vector<std::uint8_t> runCase(ProbeContext& ctx, NgxApi& ngx, Parameter* params, Handle* feature,
                                  Resources& resources, const ProbeConfig& config, int reset,
                                  bool withControlMask, const std::string& name) {
    resetCommandList(ctx);
    int result = kSuccess;
    if (ngx.useForwarder) {
        // The temporary forwarder owns the model call and its float-slot
        // workaround. Set the extra ControlMask fields here because this
        // probe keeps the forwarder's stable ABI unchanged.
        setControlMaskParams(params, resources, config.width, config.height, withControlMask);
        result = ngx.forwarderEvaluate(
            ctx.commandList.Get(), feature, params, resources.color.resource.Get(),
            resources.depth.resource.Get(), resources.motion.resource.Get(), resources.output.resource.Get(),
            config.width, config.height, config.width, config.height, 0, reset, 1.0f, 0,
            1.0f, 1.0f, 1.0f, 0, 1.0f, 1.0f);
    } else {
        setEvaluateParams(params, resources, config.width, config.height, config.width, config.height,
                          reset, 1, withControlMask);
        result = ngx.evaluateFeature(ctx.commandList.Get(), feature, params, nullptr);
    }
    checkNgx(result, ("EvaluateFeature failed for " + name).c_str());
    Readback output = createReadback(ctx, config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT);
    copyToReadback(ctx, resources.output, output);
    submitAndWait(ctx);
    auto bytes = readback(ctx, output);
    std::ofstream file(config.outputDir / (name + ".rgba16f.bin"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

double meanAbsoluteDifference(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto* pa = reinterpret_cast<const std::uint16_t*>(a.data());
    const auto* pb = reinterpret_cast<const std::uint16_t*>(b.data());
    const size_t count = a.size() / sizeof(std::uint16_t);
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += std::abs(static_cast<int>(pa[i]) - static_cast<int>(pb[i]));
    }
    return sum / static_cast<double>(count);
}

std::filesystem::path findDriverNvngx() {
    wchar_t windowsDirectory[MAX_PATH]{};
    if (GetWindowsDirectoryW(windowsDirectory, MAX_PATH) == 0) {
        fail("GetWindowsDirectoryW failed");
    }
    const auto repository = std::filesystem::path(windowsDirectory) /
                            L"System32" / L"DriverStore" / L"FileRepository";
    std::filesystem::path selected;
    std::filesystem::file_time_type selectedTime{};
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(repository, error)) {
        if (error) break;
        if (!entry.is_directory()) continue;
        const std::wstring name = entry.path().filename().wstring();
        if (name.rfind(L"nv_dispi.inf_", 0) != 0) continue;
        const auto candidate = entry.path() / L"nvngx.dll";
        if (!std::filesystem::exists(candidate)) continue;
        const auto modified = std::filesystem::last_write_time(candidate, error);
        if (error) {
            error.clear();
            continue;
        }
        if (selected.empty() || modified > selectedTime) {
            selected = candidate;
            selectedTime = modified;
        }
    }
    if (selected.empty()) fail("driver nvngx.dll not found in DriverStore");
    return selected;
}

ProbeConfig parseArgs(int argc, wchar_t** argv) {
    ProbeConfig config;
    config.root = std::filesystem::current_path();
    config.outputDir = config.root / "runtime_probe_output";
    for (int i = 1; i < argc; ++i) {
        std::wstring arg(argv[i]);
        auto value = [&](const wchar_t* prefix) -> std::wstring {
            return arg.substr(std::wcslen(prefix));
        };
        if (arg.rfind(L"--width=", 0) == 0) config.width = std::stoul(value(L"--width="));
        else if (arg.rfind(L"--height=", 0) == 0) config.height = std::stoul(value(L"--height="));
        else if (arg.rfind(L"--root=", 0) == 0) config.root = value(L"--root=");
        else if (arg.rfind(L"--output=", 0) == 0) config.outputDir = value(L"--output=");
        else if (arg.rfind(L"--driver-ngx=", 0) == 0) config.driverNvngx = value(L"--driver-ngx=");
        else if (arg == L"--backend=forwarder") config.useForwarder = true;
        else if (arg.rfind(L"--forwarder=", 0) == 0) config.forwarder = value(L"--forwarder=");
        else fail("unknown command line argument");
    }
    return config;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    const ProbeConfig config = parseArgs(argc, argv);
    std::filesystem::create_directories(config.outputDir);
    std::filesystem::create_directories(config.outputDir / L"ngx_data");
    std::filesystem::create_directories(config.outputDir / L"nr_data");
    const auto modelPath = config.root / "bin" / "nvngx_dlssnr.dll";
    if (!std::filesystem::exists(modelPath)) fail("model DLL not found under --root/bin");
    const auto forwarderPath = config.forwarder.empty()
                                   ? config.outputDir / L"nvngx.dll_dlssnr.dll"
                                   : config.forwarder;

    const auto driverNvngx = config.driverNvngx.empty() ? findDriverNvngx() : config.driverNvngx;
    if (!std::filesystem::exists(driverNvngx)) fail("driver nvngx.dll not found");

    std::cerr << "stage=create_d3d12" << std::endl;
    ProbeContext ctx = createD3D12();
    std::cerr << "stage=load_ngx" << std::endl;
    NgxApi ngx = loadNgx(driverNvngx);
    std::cerr << "stage=ngx_loaded" << std::endl;
    const auto modelDirectory = modelPath.parent_path();
    const wchar_t* searchPath = modelDirectory.c_str();
    FeatureCommonInfo commonInfo{};
    commonInfo.PathListInfo.Path = &searchPath;
    commonInfo.PathListInfo.Length = 1;
    commonInfo.LoggingInfo.MinimumLoggingLevel = 2;
    commonInfo.LoggingInfo.DisableOtherLoggingSinks = false;

    std::cerr << "stage=ngx_init" << std::endl;
    checkNgx(ngx.init(0x24480451ull, (config.outputDir / L"ngx_data").c_str(), ctx.device.Get(),
                      kVersion15, &commonInfo), "NGX D3D12 init failed");
    std::cerr << "stage=ngx_initialized" << std::endl;
    Parameter* params = nullptr;
    checkNgx(ngx.getCapabilityParameters(&params), "NGX capability parameter allocation failed");
    std::cerr << "stage=capability_params" << std::endl;
    if (!params) fail("NGX returned null capability parameters");
    if (config.useForwarder) {
        loadForwarder(ngx, forwarderPath);
        std::cout << "backend=forwarder\n";
    } else {
        const int floatSlot = discoverFloatSlot(params);
        setUInt(params, "DLSSNR.Enabled", 1);
        setUInt(params, "DLSSNR.Width", config.width);
        setUInt(params, "DLSSNR.Height", config.height);
        setUInt(params, "CreationNodeMask", 1);
        setUInt(params, "VisibilityNodeMask", 1);
        setUInt(params, "DLSSNR.Hint.Render.Preset", 0);
        setFloat(params, "DLSSNR.Intensity", 1.0f, floatSlot);
        setUInt(params, "DLSSNR.Style", 0);
        setFloat(params, "DLSSNR.LocalStructureStrength", 1.0f, floatSlot);
        setFloat(params, "DLSSNR.LocalToneStrength", 1.0f, floatSlot);
        setFloat(params, "DLSSNR.SkinStructureStrength", 1.0f, floatSlot);
        setUInt(params, "DLSSNR.UseAutoMask", 0);
        setUInt(params, "DLSSNR.UICorrection", 0);
        std::cout << "backend=driver_core\n";
    }

    resetCommandList(ctx);
    Handle* feature = nullptr;
    if (ngx.useForwarder) {
        feature = reinterpret_cast<Handle*>(ngx.forwarderCreate(
            modelPath.c_str(), (config.outputDir / L"nr_data").c_str(), ctx.device.Get(),
            ctx.commandList.Get(), params, config.width, config.height, 0, 1.0f, 0, 1.0f, 1.0f,
            1.0f, 0, 1));
        if (!feature) {
            auto lastCreate = reinterpret_cast<int*>(GetProcAddress(ngx.forwarderModule, "dlssnr_call_last_create"));
            const int result = lastCreate != nullptr ? *lastCreate : 0;
            std::cerr << "Feature 18 forwarder creation failed result=0x" << std::hex
                      << static_cast<unsigned int>(result) << std::dec << "\n";
            std::exit(3);
        }
    } else {
        checkNgx(ngx.createFeature(ctx.commandList.Get(), kFeature18, params, &feature),
                 "Feature 18 creation failed");
    }
    if (!feature) fail("Feature 18 returned null handle");
    submitAndWait(ctx);

    Resources resources;
    resources.color = createTexture(ctx, config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    resources.depth = createTexture(ctx, config.width, config.height, DXGI_FORMAT_R32_FLOAT,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    resources.motion = createTexture(ctx, config.width, config.height, DXGI_FORMAT_R16G16_FLOAT,
                                     D3D12_RESOURCE_STATE_COPY_DEST);
    resources.controlMask = createTexture(ctx, config.width, config.height, DXGI_FORMAT_R8_UNORM,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
    resources.output = createTexture(ctx, config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    std::vector<std::uint16_t> color(static_cast<size_t>(config.width) * config.height * 4);
    for (unsigned int y = 0; y < config.height; ++y) {
        for (unsigned int x = 0; x < config.width; ++x) {
            const size_t index = (static_cast<size_t>(y) * config.width + x) * 4;
            color[index + 0] = floatToHalf(static_cast<float>(x) / std::max(1u, config.width - 1));
            color[index + 1] = floatToHalf(static_cast<float>(y) / std::max(1u, config.height - 1));
            color[index + 2] = floatToHalf(0.25f);
            color[index + 3] = floatToHalf(1.0f);
        }
    }
    std::vector<float> depth(static_cast<size_t>(config.width) * config.height, 1.0f);
    std::vector<std::uint16_t> motion(static_cast<size_t>(config.width) * config.height * 2, 0);
    std::vector<std::uint8_t> mask(static_cast<size_t>(config.width) * config.height, 255);

    resetCommandList(ctx);
    uploadTexture(ctx, resources.color, config.width, config.height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                  color.data(), static_cast<size_t>(config.width) * 8);
    uploadTexture(ctx, resources.depth, config.width, config.height, DXGI_FORMAT_R32_FLOAT,
                  depth.data(), static_cast<size_t>(config.width) * 4);
    uploadTexture(ctx, resources.motion, config.width, config.height, DXGI_FORMAT_R16G16_FLOAT,
                  motion.data(), static_cast<size_t>(config.width) * 4);
    uploadTexture(ctx, resources.controlMask, config.width, config.height, DXGI_FORMAT_R8_UNORM,
                  mask.data(), config.width);
    submitAndWait(ctx);

    auto resetOutput = [&]() {
        resetCommandList(ctx);
        transition(ctx, resources.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        submitAndWait(ctx);
    };
    resetOutput();
    auto baseline = runCase(ctx, ngx, params, feature, resources, config, 1, false, "baseline_reset");
    resetOutput();
    auto temporal = runCase(ctx, ngx, params, feature, resources, config, 0, false, "temporal_zero_mv");
    resetOutput();
    auto masked = runCase(ctx, ngx, params, feature, resources, config, 0, true, "control_mask_one");

    std::fill(mask.begin(), mask.end(), 0);
    resetCommandList(ctx);
    uploadTexture(ctx, resources.controlMask, config.width, config.height, DXGI_FORMAT_R8_UNORM,
                  mask.data(), config.width);
    submitAndWait(ctx);
    resetOutput();
    auto unmasked = runCase(ctx, ngx, params, feature, resources, config, 1, true, "control_mask_zero");

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "metric baseline_vs_temporal_zero_mv=" << meanAbsoluteDifference(baseline, temporal) << "\n";
    std::cout << "metric baseline_vs_control_mask_one=" << meanAbsoluteDifference(baseline, masked) << "\n";
    std::cout << "metric baseline_vs_control_mask_zero=" << meanAbsoluteDifference(baseline, unmasked) << "\n";
    std::cout << "outputs=" << config.outputDir.string() << "\n";

    if (ngx.useForwarder) {
        ngx.forwarderRelease(feature);
    } else {
        ngx.releaseFeature(feature);
    }
    ngx.shutdown1(ctx.device.Get());
    if (ctx.fenceEvent) CloseHandle(ctx.fenceEvent);
    if (ngx.module) FreeLibrary(ngx.module);
    if (ngx.forwarderModule) FreeLibrary(ngx.forwarderModule);
    return 0;
}
