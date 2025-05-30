#include "stdafx.h"
#include "TomlConfigValueWrapper.h"

namespace
{
	void ErrorLog(const String& message) {
		Console.writeln(message);
	}

	const FilePath configPath = U"config.toml";
	const FilePath configExamplePath = U"config.example.toml";

	void assertCoincidenceWithExample(const TOMLValue& users, const TOMLValue& example) {
		for (auto&& v : users.tableView()) {
			if (not example.hasMember(v.name)) {
				ErrorLog(U"TOML parameter error: '{}' is missing in example."_fmt(v.name));
			}
			else if (v.value.getType() != example[v.name].getType()) {
				ErrorLog(U"TOML parameter error: '{}' type mismatched."_fmt(v.name));
			}
			else if (v.value.getType() == TOMLValueType::Table) {
				assertCoincidenceWithExample(v.value, example[v.name]);
			}
		}

		for (auto&& v : example.tableView()) {
			if (not users.hasMember(v.name)) {
				ErrorLog(U"TOML parameter error: '{}' is reedundant in example."_fmt(v.name));
			}
		}
	}

	struct ImplState {
		DirectoryWatcher m_directoryWatcher{U"./"};
		TOMLReader m_toml{configPath};
		TOMLReader m_exampleToml{configExamplePath};

		HashSet<String> m_erroredValues{};

		bool m_reloaded{};

		void Refresh() {
			m_reloaded = false;

			for (auto [path, action] : m_directoryWatcher.retrieveChanges()) {
				if (FileSystem::FileName(path) == U"config.toml") {
					m_toml.open(configPath);
					assertCoincidenceWithExample(m_toml, m_exampleToml);

					m_erroredValues.clear();

					m_reloaded = true;
					break;
				}
			}
		}
	};

	ImplState* s_instance{};

	void preInitialize() {
		FileSystem::Copy(configExamplePath, configPath, CopyOption::SkipExisting);
	}

	class TomlConfigValueWrapperAddon : public IAddon {
	private:
		ImplState m_state{};

	public:
		TomlConfigValueWrapperAddon() {
			if (s_instance) return;
			s_instance = &m_state;
		}

		~TomlConfigValueWrapperAddon() override {
			if (s_instance == &m_state) s_instance = nullptr;
		}

		bool init() override {
			assertCoincidenceWithExample(m_state.m_toml, m_state.m_exampleToml);
			return true;
		}

		bool update() override {
			m_state.Refresh();
			return true;
		}
	};
}

namespace Util
{
	void InitTomlConfigValueAddon() {
		preInitialize();

		Addon::Register<TomlConfigValueWrapperAddon>(U"TomlConfigValueWrapperAddon");
	}

	TOMLValue GetTomlConfigValue(const String& valuePath) {
		if (not s_instance) InitTomlConfigValueAddon();

		assert(s_instance);
		if (not s_instance) return {};
		const auto value = s_instance->m_toml[valuePath];;

		if (value.isEmpty()) {
			if (not s_instance->m_erroredValues.contains(valuePath)) {
				s_instance->m_erroredValues.emplace(valuePath);
				ErrorLog(U"TOML parameter error: '{}' is missing."_fmt(valuePath));
			}

			const auto alternative = s_instance->m_exampleToml[valuePath];
			if (not alternative.isEmpty()) return alternative;
		}

		return std::move(value);
	}

	bool IsTomlConfigHotReloaded() {
		if (not s_instance) return false;
		return s_instance->m_reloaded;
	}
}
