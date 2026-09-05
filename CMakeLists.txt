cmake_minimum_required(VERSION 3.16)
project(obs-ai-matting VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OBS REQUIRED libobs)
pkg_check_modules(ORT REQUIRED libonnxruntime)

add_library(obs-ai-matting MODULE src/plugin-main.cpp)
target_include_directories(obs-ai-matting PRIVATE /usr/include/obs ${OBS_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
target_link_directories(obs-ai-matting PRIVATE ${OBS_LIBRARY_DIRS} ${ORT_LIBRARY_DIRS})
target_link_libraries(obs-ai-matting PRIVATE ${OBS_LIBRARIES} ${ORT_LIBRARIES})
set_target_properties(obs-ai-matting PROPERTIES PREFIX "")

# Dos destinos de instalación:
#   OFF (por defecto) → el dir de plugins del usuario, sin sudo. Es el flujo del README.
#   ON               → las rutas del sistema, respetando DESTDIR/CMAKE_INSTALL_PREFIX.
#                      Lo usa el PKGBUILD; no lo actives para una instalación a mano.
option(SYSTEM_INSTALL "Instalar en las rutas del sistema (para empaquetar)" OFF)

if(SYSTEM_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS obs-ai-matting LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
        install(DIRECTORY data/
                DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/obs/obs-plugins/obs-ai-matting")
    endif()
else()
    set(PLUGIN_DEST "$ENV{HOME}/.config/obs-studio/plugins/obs-ai-matting")
    install(TARGETS obs-ai-matting LIBRARY DESTINATION "${PLUGIN_DEST}/bin/64bit")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data")
        install(DIRECTORY data/ DESTINATION "${PLUGIN_DEST}/data")
    endif()
endif()
