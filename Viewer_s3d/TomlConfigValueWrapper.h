#pragma once

namespace Util
{
	void InitTomlConfigValueAddon();

	TOMLValue GetTomlConfigValue(const String& valuePath);

	template <typename T>
	inline T GetTomlConfigValueOf(const String& valuePath) {
		return GetTomlConfigValue(String(U"config." + valuePath)).get<T>();;
	}

	template <typename T>
	inline Array<T> GetTomlDebugArrayOf(const String& valuePath) {
		Array<T> a{};
		for (const auto& v : GetTomlConfigValue(String(U"config." + valuePath)).arrayView()) {
			a.push_back(v.get<T>());
		}
		return a;;
	}

	/// @brief 現在のフレームで toml がリロードされたか
	[[nodiscard]]
	bool IsTomlConfigHotReloaded();
}
