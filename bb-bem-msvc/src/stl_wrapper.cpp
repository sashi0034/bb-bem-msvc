#include "stl_wrapper.hpp"

namespace stl_wrapper
{
    struct STLModel::Impl {
        stl_model_t m_model{};

        Impl(std::string_view filename) {
            if (!load_stl_ascii(filename.data(), &m_model)) {
                free_stl_model(&m_model);
                free(&m_model);
                throw std::runtime_error("Failed to load STL model from file: " + std::string(filename));
            }
        }

        ~Impl() {
            free_stl_model(&m_model);
            free(&m_model);
        }
    };

    STLModel::STLModel(std::string_view filename)
        : p_impl(std::make_shared<Impl>(filename)) {
    }

    std::span<const stl_facet_t> STLModel::facets() const {
        if (!p_impl) return {};
        return std::span<const stl_facet_t>(p_impl->m_model.facets, p_impl->m_model.num_facets);
    }
}
