# Add muFFT source files
set(MUFFT_SOURCES
    ${CMAKE_SOURCE_DIR}/external/muFFT/fft.c
    ${CMAKE_SOURCE_DIR}/external/muFFT/kernel.c
    ${CMAKE_SOURCE_DIR}/external/muFFT/cpu.c
)

# Add muFFT to the project
add_library(mufft STATIC ${MUFFT_SOURCES})

# Set include directories for muFFT
target_include_directories(mufft PUBLIC ${CMAKE_SOURCE_DIR}/external/muFFT)

# Enable SIMD optimizations for muFFT
target_compile_definitions(mufft PRIVATE MUFFT_USE_AVX MUFFT_USE_SSE3 MUFFT_USE_SSE)

# Link muFFT with the main target
target_link_libraries(${PROJECT_NAME} PRIVATE mufft) 