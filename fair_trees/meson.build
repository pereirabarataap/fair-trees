# fair_trees/meson.build
#
# Minimal Meson build for a “tree + utils only” scikit-learn fork.
# - Removes all top-level extension modules like _isotonic
# - Does not descend into unrelated subpackages
# - Keeps only: __check_build, utils, tree
#
# After replacing this file, delete the build dir:
#   rmdir /s /q build
#   pip install -e . --no-build-isolation

fs = import('fs')

cython_args = []

# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

# Adapted from Scipy. mingw is untested and not officially supported.
if is_mingw
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
  # Force gcc to float64 long doubles for compatibility with MSVC builds, for C only.
  add_project_arguments('-mlong-double-64', language: 'c')
endif

# -----------------------------------------------------------------------------
# NOTE: The original scikit-learn build enforces min dependency versions by
# running Python helper scripts (e.g. _min_dependencies.py). For a pruned vendor
# tree, those files may be missing, so we remove those checks here.
# -----------------------------------------------------------------------------

# NumPy include directory - needed by Cython extensions
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''
import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
'''
    ],
    check: true
  ).stdout().strip()
endif

inc_np = include_directories(incdir_numpy)

# Don't use the deprecated NumPy C API.
numpy_no_deprecated_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_no_deprecated_api)

# OpenMP (optional; build continues without it)
openmp_dep = dependency('OpenMP', language: 'c', required: false)

if not openmp_dep.found()
  warn_about_missing_openmp = true
  # On Apple Clang avoid a misleading warning if compiler variables are set.
  if host_machine.system() == 'darwin' and cc.get_id() == 'clang'
    compiler_env_vars_with_openmp = run_command(py,
      [
        '-c',
        '''
import os
compiler_env_vars_to_check = ["CPPFLAGS", "CFLAGS", "CXXFLAGS"]
compiler_env_vars_with_openmp = [
    var for var in compiler_env_vars_to_check if "-fopenmp" in os.getenv(var, "")]
print(compiler_env_vars_with_openmp)
'''
      ],
      check: true
    ).stdout().strip()
    warn_about_missing_openmp = compiler_env_vars_with_openmp == '[]'
  endif

  if warn_about_missing_openmp
    warning(
'''
                ***********
                * WARNING *
                ***********

It seems that scikit-learn cannot be built with OpenMP.
The build will continue with OpenMP-based parallelism disabled.

                    ***
''')
  else
    warning(
'''It looks like compiler environment variables were set to enable OpenMP support.
Check the output of "import fair_trees; fair_trees.show_versions()" after the build
to make sure that scikit-learn was actually built with OpenMP support.
''')
  endif
endif

# Debug Cython directives (kept for parity with upstream behavior)
boundscheck = run_command(py,
  [
    '-c',
    '''
import os
print(os.environ.get("SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES", "0") != "0")
'''
  ],
  check: true
).stdout().strip()

cython_program = find_program(cython.cmd_array()[0])

scikit_learn_cython_args = [
  '-X language_level=3',
  '-X boundscheck=' + boundscheck,
  '-X wraparound=False',
  '-X initializedcheck=False',
  '-X nonecheck=False',
  '-X cdivision=True',
  '-X profile=False',
  # Needed for cython imports across subpackages
  '--include-dir', meson.global_build_root(),
]
cython_args += scikit_learn_cython_args

# Optional: Cython >= 3.1 shared utility module (upstream behavior)
if cython.version().version_compare('>=3.1.0')
  cython_shared_src = custom_target(
    install: false,
    output: '_cyutility.c',
    command: [
      cython_program, '-3', '--fast-fail',
      '--generate-shared=' + meson.current_build_dir()/'_cyutility.c'
    ],
  )

  py.extension_module('_cyutility',
    cython_shared_src,
    subdir: 'fair_trees',
    cython_args: cython_args,
    install: true,
  )

  cython_args += ['--shared=fair_trees._cyutility']
endif

cython_gen = generator(cython_program,
  arguments: cython_args + ['@INPUT@', '--output-file', '@OUTPUT@'],
  output: '@BASENAME@.c',
)

cython_gen_cpp = generator(cython_program,
  arguments: cython_args + ['--cplus', '@INPUT@', '--output-file', '@OUTPUT@'],
  output: '@BASENAME@.cpp',
)

# Write file in Meson build dir so Python code can detect Meson build.
custom_target('write_built_with_meson_file',
  output: '_built_with_meson.py',
  command: [
    py, '-c', 'with open("fair_trees/_built_with_meson.py", "w") as f: f.write("")'
  ],
  install: true,
  install_dir: py.get_install_dir() / 'fair_trees'
)

# Need for Cython cimports across subpackages to work.
fair_trees_root_cython_tree = [
  fs.copyfile('__init__.py')
]

fair_trees_dir = py.get_install_dir() / 'fair_trees'

# -----------------------------------------------------------------------------
# Minimal subpackages required for DecisionTree / RandomForest internals
# -----------------------------------------------------------------------------
subdir('__check_build')
# subdir('ensemble') # does not require a build
# subdir('externals')
subdir('utils')
# subdir('metrics')
subdir('neighbors')
# subdir('preprocessing')
subdir('tree')


