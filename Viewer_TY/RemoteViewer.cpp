﻿#include "pch.h"

#include "imgui.h"
#include "RemoteViewer.h"

#include "LivePPAddon.h"
#include "TY/ConstantBuffer.h"
#include "TY/Graphics3D.h"
#include "TY/KeyboardInput.h"
#include "TY/Mat4x4.h"

#include "TY/Shader.h"
#include "TY/System.h"

#include "TY/Math.h"
#include "TY/Model.h"
#include "TY/ModelLoader.h"
#include "TY/RenderTarget.h"
#include "TY/Scene.h"
#include "TY/Shape3D.h"
#include "TY/Transformer3D.h"

#include "../bb-bem-msvc/src/bb_bem.h"
#include "../bb-bem-msvc/src/stl_wrapper.hpp"

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

struct Title_PointLight_impl {
    Pose m_camera{};

    Mat4x4 m_projectionMat{};

    PixelShader m_modelPS{};
    VertexShader m_modelVS{};

    ConstantBuffer<DirectionLight_cb2> m_directionLight{};

    Model m_gridPlaneModel{};

    Model m_targetModel{};

    Title_PointLight_impl() {
        resetCamera();

        const PixelShader defaultPS{ShaderParams::PS("asset/shader/model_pixel.hlsl")};
        const VertexShader defaultVS{ShaderParams::VS("asset/shader/model_vertex.hlsl")};

        m_modelPS = PixelShader{ShaderParams{.filename = shader_lambert, .entryPoint = "PS"}};
        m_modelVS = VertexShader{ShaderParams{.filename = shader_lambert, .entryPoint = "VS"}};

        const auto gridPlaneTexture = makeGridPlane(
            Size{1024, 1024}, 32, ColorF32{0.8}, ColorF32{0.9});
        m_gridPlaneModel = Model{
            ModelParams{}
            .setData(Shape3D::TexturePlane(gridPlaneTexture, Float2{100.0f, 100.0f}))
            .setShaders(defaultPS, defaultVS)
        };

        rebuildModel();
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
        m_camera.position = Float3{0.0f, 0.0f, 10.0f};
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

        constexpr double moveSpeed = 10.0f;
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

    void rebuildModel() {
        const std::string modelPath = "../input_data/torus-sd2x.stl";
        const STLModel model{modelPath};

        ModelData modelData{};

        modelData.materials.push_back({}); // TODO
        modelData.materials.back().parameters.diffuse = Float3{1.0f};

        modelData.shapes.emplace_back();
        auto& shape = modelData.shapes.back();
        shape.materialIndex = 0; // TODO: マテリアルのインデックスを設定

        for (int i = 0; i < model.facets().size(); ++i) {
            const auto& facet = model.facets()[i];

            const auto baseIndex = i * 3;
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

        m_targetModel = Model{
            ModelParams{}
            .setData(std::move(modelData))
            .setShaders(m_modelPS, m_modelVS)
            .setCB2(m_directionLight)
        };
    }
};

void Viewer_TY::RemoteViewer() {
    Title_PointLight_impl impl{};

    while (System::Update()) {
#ifdef _DEBUG
        Util::AdvanceLivePP();
#endif

        impl.Update();
    }
}
