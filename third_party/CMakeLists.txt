find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(vma INTERFACE)

add_library(stb_image INTERFACE)

add_library(tinyobjloader STATIC)

target_sources(vkbootstrap PRIVATE 
  vkbootstrap/VkBootstrap.h
  vkbootstrap/VkBootstrap.cpp
  )

# GLFW (MinGW-w64)
add_library(glfw3 STATIC IMPORTED)
set_target_properties(glfw3 PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/glfw/lib-mingw-w64/libglfw3.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/glfw/include"
)

# Volk (single-file loader)
add_library(volk STATIC
  volk/volk.c
)
target_include_directories(volk PUBLIC volk)
target_link_libraries(volk PUBLIC Vulkan::Vulkan)

target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)

#both vma and glm and header only libs so we only need the include path
target_include_directories(vma INTERFACE vma)

target_include_directories(glm INTERFACE glm)

target_sources(tinyobjloader PRIVATE 
  tinyobjloader/tiny_obj_loader.h
  tinyobjloader/tiny_obj_loader.cc
  )

target_include_directories(tinyobjloader PUBLIC tinyobjloader)

add_library(sdl2 INTERFACE)
set(sdl2_DIR "C:/SDL2" CACHE FILEPATH "Path to SDL2")

if (WIN32)
target_include_directories(sdl2 INTERFACE "${sdl2_DIR}/include/SDL2")
target_link_directories(sdl2 INTERFACE "${sdl2_DIR}/lib")
elseif (UNIX)
target_include_directories(sdl2 INTERFACE ${sdl2_DIR}/include/SDL2 ${sdl2_image_DIR}/include/SDL2)
target_link_directories(sdl2 INTERFACE ${sdl2_DIR}/lib ${sdl2_image_DIR}/lib)
endif()

target_link_libraries(sdl2 INTERFACE SDL2 SDL2main)

add_library(imgui STATIC)

target_include_directories(imgui PUBLIC imgui)

target_sources(imgui PRIVATE 
  imgui/imgui.h
  imgui/imgui.cpp

  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_widgets.cpp

  imgui/imgui_impl_vulkan.cpp
  imgui/imgui_impl_sdl.cpp
  )

target_link_libraries(imgui PUBLIC Vulkan::Vulkan sdl2)

target_include_directories(stb_image INTERFACE stb_image)
