#include "pch.h"

#include "RemoteViewer_TY.h"

#include "LivePPAddon.h"
#include "TomlConfigWrapper.h"
#include "TY/ConstantBuffer.h"
#include "TY/Graphics3D.h"
#include "TY/KeyboardInput.h"
#include "TY/Mat4x4.h"

#include "TY/Shader.h"
#include "TY/System.h"

#include "TY/Math.h"
#include "TY/ModelDrawer.h"
#include "TY/ModelLoader.h"
#include "TY/RenderTarget.h"
#include "TY/Scene.h"
#include "TY/Shape3D.h"
#include "TY/Transformer3D.h"

#include "../bb-bem-msvc/src/bb_bem.h"
#include "../bb-bem-msvc/src/stl_wrapper.hpp"
#include "../Viewer_s3d/bb_bem_wrapper.h"
#include "TY/HSV.h"

using namespace TY;

namespace
{
    struct DirectionLight_cb2 {
        alignas(16) Float3 lightDirection;
        alignas(16) Float3 lightColor{};
    };

    struct Pose {
        Float3 position{};
        Float3 rotation{}; // Euler angles in radians

        Mat4x4 getMatrix() const {
            return Mat4x4::Identity()
                   .rotatedX(rotation.x)
                   .rotatedY(rotation.y)
                   .rotatedZ(rotation.z)
                   .translated(position);
            // return Mat4x4::RollPitchYaw(rotation).translated(position);
        }
    };

    ShaderResourceTexture makeGridPlane(
        const Size& size, int lineSpacing, const UnifiedColor& lineColor, const UnifiedColor& backColor) {
        Image image{size, backColor};
        const Size padding = (size % lineSpacing) / 2;

        for (int x = padding.x; x < size.x; x += lineSpacing) {
            for (int y = 0; y < size.y; y++) {
                image[Point{x, y}] = lineColor;
            }
        }

        for (int y = padding.y; y < size.y; y += lineSpacing) {
            for (int x = 0; x < size.x; x++) {
                image[Point{x, y}] = lineColor;
            }
        }

        return ShaderResourceTexture{image};
    }

    const std::string shader_lambert = "asset/shader/lambert.hlsl";
}

struct RemoteViewer_TY {
    Pose m_camera{};

    Mat4x4 m_projectionMat{};

    PixelShader m_modelPS{};
    VertexShader m_modelVS{};

    ConstantBuffer<DirectionLight_cb2> m_directionLight{};

    ModelDrawer m_gridPlaneModel{};

    ModelDrawer m_targetModel{};

    bb_bem_wrapper::ResultHolder m_bb{};

    bb_compute_t m_currentCompute{};
    int8_t m_currentBatch{};

    RemoteViewer_TY() {
        resetCamera();

        const PixelShader defaultPS{ShaderParams::PS("asset/shader/model_pixel.hlsl")};
        const VertexShader defaultVS{ShaderParams::VS("asset/shader/model_vertex.hlsl")};

        m_modelPS = PixelShader{ShaderParams{.filepath = shader_lambert, .entryPoint = "PS"}};
        m_modelVS = VertexShader{ShaderParams{.filepath = shader_lambert, .entryPoint = "VS"}};
        // m_modelPS = defaultPS;
        // m_modelVS = defaultVS;

        const auto gridPlaneTexture = makeGridPlane(
            Size{1024, 1024}, 32, ColorF32{0.8}, ColorF32{0.9});
        m_gridPlaneModel = ModelDrawer{
            ModelDrawerParams{}
            .setModel(Shape3D::TexturePlane(gridPlaneTexture, Float2{100.0f, 100.0f}))
            .setShaders(defaultPS, defaultVS)
        };

        calculate_bem();
    }

    void Update() {
        if (not KeyShift.pressed()) {
            updateCamera();
        }

        m_directionLight->lightDirection = m_camera.getMatrix().forward().normalized();
        m_directionLight->lightColor = Float3{1.0f};
        m_directionLight.upload();

        {
            Pose pose{};
            pose.position.y = -10.0f;
            const Transformer3D t3d{pose.getMatrix()};
            m_gridPlaneModel.draw();
        }

        {
            m_targetModel.draw();
        }

        {
            ImGui::Begin("Camera Info");

            ImGui::Text("Position: (%.2f, %.2f, %.2f)",
                        m_camera.position.x,
                        m_camera.position.y,
                        m_camera.position.z);

            ImGui::Text("Rotation (rad): (%.2f, %.2f, %.2f)",
                        m_camera.rotation.x,
                        m_camera.rotation.y,
                        m_camera.rotation.z);

            ImGui::Text("Rotation (deg): (%.1f, %.1f, %.1f)",
                        Math::ToDegrees(m_camera.rotation.x),
                        Math::ToDegrees(m_camera.rotation.y),
                        Math::ToDegrees(m_camera.rotation.z));

            ImGui::Text("Light Direction: (%.2f, %.2f, %.2f)",
                        m_directionLight->lightDirection.x,
                        m_directionLight->lightDirection.y,
                        m_directionLight->lightDirection.z);

            ImGui::End();
        }

        {
            ImGui::Begin("System Settings");

            static bool s_sleep{};;
            ImGui::Checkbox("Sleep", &s_sleep);

            if (s_sleep) {
                System::Sleep(500);
            }

            ImGui::End();
        }
    }

    void resetCamera() {
        m_camera.position = Float3{0.0f, 0.0f, 1.0f};
        m_camera.rotation = Float3{-Math::PiF / 4.0f, 0.0f, 0.0f};
    }

    static Pose getPoseInput() {
        Pose pose{};

        pose.position.x = (KeyD.pressed() ? 1.0f : 0.0f) - (KeyA.pressed() ? 1.0f : 0.0f);
        pose.position.y = (KeyE.pressed() ? 1.0f : 0.0f) - (KeyX.pressed() ? 1.0f : 0.0f);
        pose.position.z = (KeyW.pressed() ? 1.0f : 0.0f) - (KeyS.pressed() ? 1.0f : 0.0f);

        pose.rotation.x = (KeyRight.pressed() ? 1.0f : 0.0f) - (KeyLeft.pressed() ? 1.0f : 0.0f);
        pose.rotation.y = (KeyDown.pressed() ? 1.0f : 0.0f) - (KeyUp.pressed() ? 1.0f : 0.0f);

        return pose;
    }

    void updateCamera() {
        if (KeyR.down()) {
            resetCamera();
        }

        const auto poseInput = getPoseInput();
        const Float3 moveVector = poseInput.position.normalized();
        const Float3 rotateVector = poseInput.rotation.normalized();

        constexpr double moveSpeed = 1.0f;
        constexpr double rotationSpeed = 50.0f;

        if (not moveVector.isZero()) {
            m_camera.position += -moveVector * moveSpeed * System::DeltaTime();
        }

        if (not rotateVector.isZero()) {
            m_camera.rotation.x += Math::ToRadians(-rotateVector.y * rotationSpeed * System::DeltaTime());
            m_camera.rotation.y += Math::ToRadians(rotateVector.x * rotationSpeed * System::DeltaTime());
        }

        Graphics3D::SetViewMatrix(m_camera.getMatrix());

        m_projectionMat = Mat4x4::PerspectiveFov(
            90.0_deg,
            Scene::Size().horizontalAspectRatio(),
            0.1f,
            100.0f
        );

        Graphics3D::SetProjectionMatrix(m_projectionMat);
    }

    void calculate_bem() {
        m_bb.release();

        if (not calculate_bem_internal()) {
            std::cerr << "Error: Boundary element analysis failed." << std::endl;
            return;
        }

        m_bb.varifyResult();
        rebuildModel();
    }

    bool calculate_bem_internal() {
        const std::string filename = Viewer_TY::GetTomlConfigValueAsPath("input_path");

        if (bb_bem(filename.data(), BB_COMPUTE_NAIVE, &m_bb.bb_naive) != BB_OK) return false;
        if (bb_bem(filename.data(), BB_COMPUTE_CUDA, &m_bb.bb_cuda) != BB_OK) return false;
        if (bb_bem(filename.data(), BB_COMPUTE_CUDA_WMMA, &m_bb.bb_cuda_wmma) != BB_OK) return false;

        return true;
    }

    void rebuildModel() {
        const std::string modelPath = Viewer_TY::GetTomlConfigValueAsPath("input_path");
        const STLModel model{modelPath};

        // 最大値を求める
        const auto& bb_result = m_bb.get(m_currentCompute);
        const auto& bb_input = bb_result.input;
        double maxSolAbs{};
        for (int fc_id = 0; fc_id < bb_input.nofc_unaligned; ++fc_id) {
            const double sol = bb_result.sol[fc_id][m_currentBatch];
            maxSolAbs = Max(maxSolAbs, Abs(sol));
        }

        if (maxSolAbs == 0.0) maxSolAbs = 1.0;

        // -----------------------------------------------

        ModelData modelData{};

        // マテリアル生成: 正
        constexpr int colorResolution = 64;
        for (int i = 0; i < colorResolution; ++i) {
            auto color = HSV{ColorF32{0.97, 0.29, 0}};
            color.s = (i + 1.0f) / static_cast<float>(colorResolution);

            modelData.materials.push_back({});
            modelData.materials.back().parameters.diffuse = color.toColorF().toFloat3();
        }

        // マテリアル生成: 負
        for (int i = 0; i < colorResolution; ++i) {
            auto color = HSV{ColorF32{0.18, 0.35, 0.85}};
            color.s = (i + 1.0f) / static_cast<float>(colorResolution);

            modelData.materials.push_back({});
            modelData.materials.back().parameters.diffuse = color.toColorF().toFloat3();
        }

        modelData.shapes.resize(modelData.materials.size());
        for (int i = 0; i < modelData.shapes.size(); ++i) {
            modelData.shapes[i].materialIndex = i;
        }

        // シェイプ生成
        for (int fc_id = 0; fc_id < bb_input.nofc_unaligned; ++fc_id) {
            const double sol = bb_result.sol[fc_id][m_currentBatch];
            const int shapeIndex = (Abs(sol) / maxSolAbs) * colorResolution + (sol < 0.0 ? colorResolution : 0);
            auto& shape = modelData.shapes[shapeIndex];

            const auto& facet = model.facets()[fc_id];

            const auto baseIndex = shape.vertexBuffer.size();
            shape.indexBuffer.push_back(baseIndex);
            shape.indexBuffer.push_back(baseIndex + 1);
            shape.indexBuffer.push_back(baseIndex + 2);

            for (int j = 0; j < 3; ++j) {
                const auto& vertex = facet.v[j];

                ModelVertex modelVertex{};
                modelVertex.position = Float3{vertex.x, vertex.y, vertex.z};
                modelVertex.normal = Float3{facet.normal.x, facet.normal.y, facet.normal.z};

                shape.vertexBuffer.push_back(modelVertex);
            }
        }

        // 空のマテリアルを削除
        for (int i = modelData.materials.size() - 1; i >= 0; --i) {
            if (modelData.shapes[i].vertexBuffer.empty()) {
                modelData.shapes.remove_at(i);
                modelData.materials.remove_at(i);

                // 後ろにあるマテリアルのインデックスを落とす
                for (int j = i; j < modelData.shapes.size(); ++j) {
                    modelData.shapes[j].materialIndex--;
                }
            }
        }

        m_targetModel = ModelDrawer{
            ModelDrawerParams{}
            .setData(std::move(modelData))
            .setShaders(m_modelPS, m_modelVS)
            .setCB2(m_directionLight)
        };
    }
};

void Viewer_TY::RemoteViewer() {
    RemoteViewer_TY impl{};

    while (System::Update()) {
#ifdef _DEBUG
        Util::AdvanceLivePP();
#endif
        Viewer_TY::AdvanceTomlConfig();

        impl.Update();
    }
}
