// Capture the graphics-side ABI around NGX/DLSS5 without modifying commands.
// This intentionally logs handles, dimensions, layouts, push constants and
// dispatches only. It is a diagnostic add-on for a disposable ReShade runtime.

#include <windows.h>
#include <d3d12.h>
#include <reshade.hpp>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

using namespace reshade::api;

namespace {

std::mutex g_mutex;
std::ofstream g_log;

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
    g_log.open(full, std::ios::out | std::ios::trunc);
}

template <typename F>
void log(F &&make_line) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_log.is_open()) open_log();
    g_log << make_line() << '\n';
    g_log.flush();
}

void on_init_device(device *dev) {
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
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
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
        reshade::unregister_addon(hModule);
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_log.is_open()) g_log.close();
    }
    return TRUE;
}
