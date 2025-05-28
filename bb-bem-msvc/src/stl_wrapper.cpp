#include "stl_wrapper.hpp"

namespace stl_wrapper
{
    struct STLModel::Impl {
        bool m_initialized{};

        stl_model_t m_model{};

        Impl(std::string_view filename) {
            if (!load_stl_ascii(filename.data(), &m_model)) {
                free_stl_model(&m_model);
                return;
            }

            m_initialized = true;
        }

        ~Impl() {
            free_stl_model(&m_model);
        }
    };

    STLModel::STLModel(std::string_view filename)
        : p_impl(std::make_shared<Impl>(filename)) {
    }

    bool STLModel::isValid() const {
        return p_impl && p_impl->m_initialized;
    }

    std::span<const stl_facet_t> STLModel::facets() const {
        if (!p_impl) return {};
        return std::span<const stl_facet_t>(p_impl->m_model.facets, p_impl->m_model.num_facets);
    }
}
