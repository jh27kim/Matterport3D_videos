#ifndef Magnum_BulletIntegration_DebugDraw_h
#define Magnum_BulletIntegration_DebugDraw_h
/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>
    Copyright © 2016 Jonathan Hale <squareys@googlemail.com>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

/** @file
 * @brief Implementation of `btIDebugDraw` for physics visualization in Bullet
 */

#include <vector>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Macros.h>
#include <LinearMath/btIDebugDraw.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/VertexColor.h>

#include "Magnum/BulletIntegration/Integration.h"

namespace Magnum { namespace BulletIntegration {

/**
@brief Bullet physics debug visualization

This class implements `btIDebugDraw`, which allows rendering a visualization
of a Bullet physics world for debugging purposes.

@section BulletIntegration-DebugDraw-usage Usage

Set up an instance and attach it to Bullet world:

@snippet BulletIntegration.cpp DebugDraw-usage

Then, at every frame, call this:

@snippet BulletIntegration.cpp DebugDraw-usage-per-frame
*/
class MAGNUM_BULLETINTEGRATION_EXPORT DebugDraw: public btIDebugDraw {
    public:
        /**
         * @brief Debug mode
         *
         * @see @ref Modes, @ref setMode()
         */
        enum class Mode: Int {
            /** Disable debug rendering */
            NoDebug = DBG_NoDebug,

            /** Draw wireframe of all collision shapes */
            DrawWireframe = DBG_DrawWireframe,

            /** Draw axis aligned bounding box of all collision object */
            DrawAabb = DBG_DrawAabb,

            /** Draw text for features */
            DrawFeaturesText = DBG_DrawFeaturesText,

            /** Draw contact points */
            DrawContactPoints = DBG_DrawContactPoints,

            /** Disable deactivation of objects */
            NoDeactivation = DBG_NoDeactivation,

            /** Diable help text */
            NoHelpText = DBG_NoHelpText,

            /** Enable text drawing */
            DrawText = DBG_DrawText,

            /** Profile timings */
            ProfileTimings = DBG_ProfileTimings,

            /** Enable Sat Comparison */
            EnableSatComparison = DBG_EnableSatComparison,

            /** Disable Bullet LCP */
            DisableBulletLCP = DBG_DisableBulletLCP,

            /** Enable CCD */
            EnableCCD = DBG_EnableCCD,

            /** Draw constaints */
            DrawConstraints = DBG_DrawConstraints,

            /** Draw constraint limits */
            DrawConstraintLimits = DBG_DrawConstraintLimits,

            /** Draw fast wireframes */
            FastWireframe = DBG_FastWireframe,

            #ifdef MAGNUM_BUILD_DEPRECATED
            /** Draw fast wireframes
             * @deprecated Use @ref Mode::FastWireframe instead.
             */
            FastWirefram CORRADE_DEPRECATED_ENUM("use FastWireframe instead") = FastWireframe,
            #endif

            /** Draw normals */
            DrawNormals = DBG_DrawNormals,

            #if BT_BULLET_VERSION >= 284
            /**
             * Draw frames
             *
             * @note Supported since Bullet 2.83.5.
             */
            DrawFrames = DBG_DrawFrames
            #endif
        };

        #ifdef MAGNUM_BUILD_DEPRECATED
        /** @brief @copybrief Mode
         * @deprecated Use @ref Mode instead.
         */
        typedef CORRADE_DEPRECATED("use Mode instead") Mode DebugMode;
        #endif

        /**
         * @brief Debug modes
         *
         * @see @ref setMode()
         */
        typedef Containers::EnumSet<Mode> Modes;

        #ifdef MAGNUM_BUILD_DEPRECATED
        /** @brief @copybrief Modes
         * @deprecated Use @ref Modes instead.
         */
        typedef CORRADE_DEPRECATED("use Modes instead") Modes DebugModes;
        #endif

        /**
         * @brief Constructor
         * @param initialBufferCapacity     Amount of lines for which to
         *      reserve memory in the buffer vector.
         *
         * Sets up @ref Shaders::VertexColor3D, @ref GL::Buffer and
         * @ref GL::Mesh for physics debug rendering.
         */
        explicit DebugDraw(std::size_t initialBufferCapacity = 0);

        /**
         * @brief Construct without creating the underlying OpenGL objects
         *
         * The constructed instance is equivalent to moved-from state. Useful
         * for deferring the initialization to a later point, for example if
         * the OpenGL context is not yet created. Move another instance over it
         * to make it useful.
         */
        explicit DebugDraw(NoCreateT) noexcept;

        /** @brief Move constructor */
        DebugDraw(DebugDraw&&) noexcept;

        /** @brief Copying is not allowed */
        DebugDraw(const DebugDraw&) = delete;

        ~DebugDraw();

        /** @brief Move assignment */
        DebugDraw& operator=(DebugDraw&&) noexcept;

        /** @brief Copying is not allowed */
        DebugDraw& operator=(const DebugDraw&) = delete;

        /** @brief Debug mode */
        Modes mode() const { return _mode; }

        #ifdef MAGNUM_BUILD_DEPRECATED
        /** @brief @copybrief mode()
         * @deprecated Use @ref mode() instead.
         */
        CORRADE_DEPRECATED("use mode() instead") Modes debugMode() const {
            return mode();
        }
        #endif

        /**
         * @brief Set debug mode
         * @return Reference to self (for method chaining)
         *
         * By default, nothing is enabled.
         */
        DebugDraw& setMode(Modes mode) {
            _mode = mode;
            return *this;
        }

        #ifdef MAGNUM_BUILD_DEPRECATED
        /** @brief @copybrief setMode()
         * @deprecated Use @ref setMode() instead.
         */
        CORRADE_DEPRECATED("use setMode() instead") DebugDraw& setDebugMode(Modes mode) {
            return setMode(mode);
        }
        #endif

        /** @brief Set transformation projection matrix used for rendering */
        DebugDraw& setTransformationProjectionMatrix(const Matrix4& matrix) {
            _transformationProjectionMatrix = matrix;
            return *this;
        }

    private:
        void setDebugMode(int debugMode) override;
        int getDebugMode() const override;
        void drawLine(const btVector3& from, const btVector3& to, const btVector3& color) override;
        void drawLine(const btVector3& from, const btVector3& to, const btVector3& fromColor, const btVector3& toColor) override;
        void drawContactPoint(const btVector3& pointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) override;
        void reportErrorWarning(const char *warningString) override;
        void draw3dText(const btVector3& location, const char* textString) override;
        void flushLines()
            /* See comment in drawLine() for detailed rant */
            #if BT_BULLET_VERSION >= 284
            override
            #endif
            ;

        Modes _mode{};

        Matrix4 _transformationProjectionMatrix;
        Shaders::VertexColor3D _shader;

        GL::Buffer _buffer;
        GL::Mesh _mesh;
        std::vector<Vector3> _bufferData;
};

CORRADE_ENUMSET_OPERATORS(DebugDraw::Modes)

/** @debugoperatorenum{Magnum::BulletIntegration::DebugDraw::Mode} */
MAGNUM_BULLETINTEGRATION_EXPORT Debug& operator<<(Debug& debug, DebugDraw::Mode value);

}}

#endif
