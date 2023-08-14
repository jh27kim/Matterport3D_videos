if "%APPVEYOR_BUILD_WORKER_IMAGE%" == "Visual Studio 2017" call "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvarsall.bat" x64 || exit /b
if "%APPVEYOR_BUILD_WORKER_IMAGE%" == "Visual Studio 2015" call "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/vcvarsall.bat" x64 || exit /b
set PATH=%APPVEYOR_BUILD_FOLDER%/openal/bin/Win64;%APPVEYOR_BUILD_FOLDER%\deps\bin;%PATH%

rem Build Corrade
git clone --depth 1 git://github.com/mosra/corrade.git || exit /b
cd corrade || exit /b
mkdir build && cd build || exit /b
cmake .. ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_INSTALL_PREFIX=%APPVEYOR_BUILD_FOLDER%/deps ^
    -DWITH_INTERCONNECT=OFF ^
    -DUTILITY_USE_ANSI_COLORS=ON ^
    -G Ninja || exit /b
cmake --build . || exit /b
cmake --build . --target install || exit /b
cd .. && cd ..

rem Build the fastest Vulkan driver ever. See appveyor.yml for why Vulkan is
rem a separate build for now.
cl.exe /c package/ci/libvulkan.cpp || exit /b
lib.exe /OUT:%APPVEYOR_BUILD_FOLDER%/deps/lib/libvulkan.lib libvulkan.obj || exit /b

rem Enabling only stuff that's directly affected by Vulkan, disabling
rem everything else.
mkdir build && cd build || exit /b
cmake .. ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_INSTALL_PREFIX=%APPVEYOR_BUILD_FOLDER%/deps ^
    -DVulkan_LIBRARY=%APPVEYOR_BUILD_FOLDER%/deps/lib/libvulkan.lib ^
    -DWITH_AUDIO=OFF ^
    -DWITH_DEBUGTOOLS=OFF ^
    -DWITH_GL=OFF ^
    -DWITH_MESHTOOLS=OFF ^
    -DWITH_PRIMITIVES=OFF ^
    -DWITH_SCENEGRAPH=OFF ^
    -DWITH_SHADERS=OFF ^
    -DWITH_TEXT=OFF ^
    -DWITH_TEXTURETOOLS=OFF ^
    -DWITH_TRADE=OFF ^
    -DWITH_VK=ON ^
    -DWITH_ANYAUDIOIMPORTER=OFF ^
    -DWITH_ANYIMAGECONVERTER=OFF ^
    -DWITH_ANYIMAGEIMPORTER=OFF ^
    -DWITH_ANYSCENEIMPORTER=OFF ^
    -DWITH_MAGNUMFONT=OFF ^
    -DWITH_MAGNUMFONTCONVERTER=OFF ^
    -DWITH_OBJIMPORTER=OFF ^
    -DWITH_TGAIMAGECONVERTER=OFF ^
    -DWITH_TGAIMPORTER=OFF ^
    -DWITH_WAVAUDIOIMPORTER=OFF ^
    -DWITH_DISTANCEFIELDCONVERTER=OFF ^
    -DWITH_FONTCONVERTER=OFF ^
    -DWITH_IMAGECONVERTER=OFF ^
    -DWITH_GL_INFO=OFF ^
    -DWITH_AL_INFO=OFF ^
    -DBUILD_TESTS=ON ^
    -DBUILD_GL_TESTS=OFF ^
    -G Ninja || exit /b
cmake --build . || exit /b
cmake --build . --target install || exit /b

rem Test
set CORRADE_TEST_COLOR=ON
ctest -V -E GLTest || exit /b
