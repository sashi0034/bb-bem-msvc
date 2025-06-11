#include "pch.h"
#include "TomlConfigWrapper.h"

#include "TY/FileWatcher.h"

namespace
{
    const std::string configaPath = "../Viewer_s3d/App/config.toml";

    bool s_initialized = false;

    toml::parse_result s_toml{};

    TY::FileWatcher s_fileWatcher{};

    void ensureLoaded() {
        if (not s_initialized) {
            s_toml = toml::parse_file(configaPath);

            s_fileWatcher = TY::FileWatcher{configaPath};

            s_initialized = true;
        }
    }
}

namespace Viewer_TY
{
    void AdvanceTomlConfig() {
        ensureLoaded();

        if (s_fileWatcher.isChangedInFrame()) {
            s_toml = toml::parse_file(configaPath);
        }
    }

    std::string GetTomlConfigDirectory() {
        return std::filesystem::path(configaPath).parent_path().string();
    }

    toml::node_view<toml::node> GetTomlConfigValue(const std::string& valuePath) {
        ensureLoaded();

        return s_toml.at_path(valuePath);
    }

    std::string GetTomlConfigValueAsPath(const std::string& valuePath) {
        const auto path = GetTomlConfigValueOf<std::string>(valuePath);
        namespace fs = std::filesystem;

        if (fs::path(path).is_absolute()) {
            return path;
        } else {
            return (fs::path(GetTomlConfigDirectory()) / path).string();
        }
    }
}
