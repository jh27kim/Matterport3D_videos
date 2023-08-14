EAPI=5

EGIT_REPO_URI="git://github.com/mosra/magnum-plugins.git"

inherit cmake-utils git-r3

DESCRIPTION="Plugins for the Magnum C++11/C++14 graphics engine"
HOMEPAGE="https://magnum.graphics"

LICENSE="MIT"
SLOT="0"
KEYWORDS="~amd64 ~x86"
IUSE=""

RDEPEND="
	dev-libs/magnum
	media-libs/devil
	media-libs/faad2
	media-libs/freetype
	media-libs/harfbuzz
	virtual/jpeg
	media-libs/libpng
	media-libs/assimp
"
DEPEND="${RDEPEND}"

src_configure() {
	# general configuration
	local mycmakeargs=(
		-DCMAKE_INSTALL_PREFIX="${EPREFIX}/usr"
		-DCMAKE_BUILD_TYPE=Release
		-DWITH_ASSIMPIMPORTER=ON
		-DWITH_DDSIMPORTER=ON
		-DWITH_DEVILIMAGEIMPORTER=ON
		-DWITH_DRFLACAUDIOIMPORTER=ON
		-DWITH_DRWAVAUDIOIMPORTER=ON
		-DWITH_FAAD2AUDIOIMPORTER=ON
		-DWITH_FREETYPEFONT=ON
		-DWITH_HARFBUZZFONT=ON
		-DWITH_JPEGIMAGECONVERTER=ON
		-DWITH_JPEGIMPORTER=ON
		-DWITH_MINIEXRIMAGECONVERTER=ON
		-DWITH_OPENGEXIMPORTER=ON
		-DWITH_PNGIMAGECONVERTER=ON
		-DWITH_PNGIMPORTER=ON
		-DWITH_STANFORDIMPORTER=ON
		-DWITH_STBIMAGECONVERTER=ON
		-DWITH_STBIMAGEIMPORTER=ON
		-DWITH_STBTRUETYPEFONT=ON
		-DWITH_STBVORBISAUDIOIMPORTER=ON
		-DWITH_TINYGLTFIMPORTER=ON
	)
	cmake-utils_src_configure
}

# kate: replace-tabs off;
