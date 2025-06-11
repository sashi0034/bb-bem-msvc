#pragma once

namespace Viewer_TY
{
    void AdvanceTomlConfig();

    std::string GetTomlConfigDirectory();

    toml::node_view<toml::node> GetTomlConfigValue(const std::string& valuePath);

    template <typename T>
    inline T GetTomlConfigValueOf(const std::string& valuePath) {
        if (const auto v = GetTomlConfigValue("config." + valuePath).value<T>()) {
            return *v;
        } else {
            return T();
        }
    }

    std::string GetTomlConfigValueAsPath(const std::string& valuePath);
}
