#ifndef Magnum_Trade_PngImporter_h
#define Magnum_Trade_PngImporter_h
/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>

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
 * @brief Class @ref Magnum::Trade::PngImporter
 */

#include <Corrade/Containers/Array.h>
#include <Magnum/Trade/AbstractImporter.h>

#include "MagnumPlugins/PngImporter/configure.h"

#ifndef DOXYGEN_GENERATING_OUTPUT
#ifndef MAGNUM_PNGIMPORTER_BUILD_STATIC
    #ifdef PngImporter_EXPORTS
        #define MAGNUM_PNGIMPORTER_EXPORT CORRADE_VISIBILITY_EXPORT
    #else
        #define MAGNUM_PNGIMPORTER_EXPORT CORRADE_VISIBILITY_IMPORT
    #endif
#else
    #define MAGNUM_PNGIMPORTER_EXPORT CORRADE_VISIBILITY_STATIC
#endif
#define MAGNUM_PNGIMPORTER_LOCAL CORRADE_VISIBILITY_LOCAL
#else
#define MAGNUM_PNGIMPORTER_EXPORT
#define MAGNUM_PNGIMPORTER_LOCAL
#endif

namespace Magnum { namespace Trade {

/**
@brief PNG importer plugin

Supports Portable Network Graphics (`*.png`) RGB, RGBA or grayscale images with
8 and 16 bits per channel. Palleted images and images with transparency mask
are automatically converted to RGB(A).

This plugin depends on the @ref Trade and [libPNG](http://www.libpng.org/pub/png/libpng.html)
libraries and is built if `WITH_PNGIMPORTER` is enabled when building Magnum
Plugins. To use as a dynamic plugin, you need to load the @cpp "PngImporter" @ce
plugin from `MAGNUM_PLUGINS_IMPORTER_DIR`. To use as a static plugin or as a
dependency of another plugin with CMake, you need to request the `PngImporter`
component of the `MagnumPlugins` package and link to the
`MagnumPlugins::PngImporter` target. See @ref building-plugins,
@ref cmake-plugins and @ref plugins for more information.

@m_class{m-block m-success}

@thirdparty This plugin makes use of the
    [libPNG](http://www.libpng.org/pub/png/libpng.html) library, released under
    the @m_class{m-label m-success} **libPNG** license
    ([license text](http://libpng.org/pub/png/src/libpng-LICENSE.txt)). It
    requires attribution for public use.

@section Trade-PngImporter-limitations Behavior and limitations

The images are imported with @ref PixelFormat::RGB8Unorm /
@ref PixelFormat::RGB16Unorm, @ref PixelFormat::RGBA8Unorm /
@ref PixelFormat::RGBA16Unorm or @ref PixelFormat::R8Unorm /
@ref PixelFormat::R16Unorm. All imported images use default @ref PixelStorage
parameters.

@subsection Trade-PngImporter-limitations-cgbi Apple CgBI PNGs

CgBI is a proprietary Apple-specific extension to PNG
([details here](http://iphonedevwiki.net/index.php/CgBI_file_format)). Stock
`libPNG` doesn't support these, you might want to use some fork that supports
it, for example https://github.com/jongwook/libpng. Or use @ref StbImageImporter,
which has the support.

The test for this plugin contains a file that can be used for verifying CgBI
support.
*/
class MAGNUM_PNGIMPORTER_EXPORT PngImporter: public AbstractImporter {
    public:
        /** @brief Default constructor */
        explicit PngImporter();

        /** @brief Plugin manager constructor */
        explicit PngImporter(PluginManager::AbstractManager& manager, const std::string& plugin);

        ~PngImporter();

    private:
        MAGNUM_PNGIMPORTER_LOCAL Features doFeatures() const override;
        MAGNUM_PNGIMPORTER_LOCAL bool doIsOpened() const override;
        MAGNUM_PNGIMPORTER_LOCAL void doClose() override;
        MAGNUM_PNGIMPORTER_LOCAL void doOpenData(Containers::ArrayView<const char> data) override;

        MAGNUM_PNGIMPORTER_LOCAL UnsignedInt doImage2DCount() const override;
        MAGNUM_PNGIMPORTER_LOCAL Containers::Optional<ImageData2D> doImage2D(UnsignedInt id) override;

        Containers::Array<unsigned char> _in;
};

}}

#endif
