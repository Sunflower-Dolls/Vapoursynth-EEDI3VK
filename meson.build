project('eedi3vk', 'cpp',
  version : '1.2',
  license : 'GPL-3.0-or-later',
  default_options : [
    'cpp_std=c++23',
    'warning_level=3',
    'buildtype=release'
  ]
)

cxx = meson.get_compiler('cpp')

buildtype = get_option('buildtype')
if buildtype != 'debug' and buildtype != 'debugoptimized'
  add_project_link_arguments('-flto', language: ['cpp'])
  add_project_arguments('-flto', language: ['cpp'])
endif

vapoursynth_dep = dependency('vapoursynth', version: '>=55')

vulkan_headers_dep = dependency(
  'vulkan-headers',
  fallback: ['vulkan-headers', 'vulkan_headers_dep'],
  required: false,
)

# Some distros don't ship a standalone `vulkan-headers.pc`.
# Try the system Vulkan SDK's aggregate dependency instead.
if not vulkan_headers_dep.found()
  vulkan_dep = dependency('vulkan', required: false)
  if vulkan_dep.found()
    vulkan_headers_dep = vulkan_dep.partial_dependency(
      includes: true,
      compile_args: true
    )
  else
    vulkan_headers_dep = dependency(
      'vulkan-headers',
      fallback: ['vulkan-headers', 'vulkan_headers_dep']
    )
  endif
endif

volk_dep = dependency('volk',
  fallback: ['volk', 'volk_dep']
)

vma_dep = dependency('vulkan_memory_allocator',
  fallback: ['vulkan-memory-allocator', 'vma_dep']
)

dependencies = [
  vapoursynth_dep,
  vulkan_headers_dep,
  volk_dep,
  vma_dep
]

install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'

sources = [
  'src/eedi3vk.cpp',
  'src/vulkan/VulkanContext.cpp',
  'src/vulkan/VulkanMemory.cpp',
  'src/vulkan/VulkanComputePipeline.cpp',
  'src/vulkan/VulkanResourcePool.cpp',
  'src/vulkan/EEDI3Pipelines.cpp',
]

glslc = find_program('glslc', required: true)

shader_files = [
  'src/shaders/copy_field.comp',
  'src/shaders/dilate_mask.comp',
  'src/shaders/calc_costs.comp',
  'src/shaders/viterbi_scan.comp',
  'src/shaders/interpolate.comp',
  'src/shaders/copy_buffer.comp',
]

spirv_outputs = []
foreach shader : shader_files
  shader_name = shader.split('/')[-1].split('.')[0]
  spirv_target = custom_target(shader_name + '_spv',
    input: shader,
    output: shader_name + '.spv',
    command: [glslc, '--target-env=vulkan1.1', '-O', '@INPUT@', '-o', '@OUTPUT@'],
    build_by_default: true
  )
  spirv_outputs += spirv_target
endforeach

spirv_include = include_directories('.')

eedi3vk_module = shared_module('eedi3vk', sources,
  dependencies: dependencies,
  include_directories: spirv_include,
  link_depends: spirv_outputs,
  install: true,
  install_dir: install_dir
)

if host_machine.system() == 'darwin'
  custom_target('eedi3vk_dsym',
    input: eedi3vk_module,
    output: 'libeedi3vk.dylib.dSYM',
    command: ['dsymutil', '@INPUT@'],
    build_by_default: true
  )
endif
