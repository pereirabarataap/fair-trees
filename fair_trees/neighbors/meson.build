# fair_trees/neighbors/meson.build
# Minimal neighbors build: only compile the _quad_tree extension needed by tree.

fs = import('fs')

# Ensure the package is installed (pure-python files)
neighbors_dir = py.get_install_dir() / 'fair_trees' / 'neighbors'
# install_subdir('.', install_dir: neighbors_dir)

# Cython tree needed for local cimports
neighbors_cython_tree = [
  fs.copyfile('__init__.py'),
  fs.copyfile('_quad_tree.pxd'),
]

py.extension_module(
  '_quad_tree',
  [cython_gen.process('_quad_tree.pyx'), neighbors_cython_tree],
  dependencies: [np_dep],
  cython_args: cython_args,
  subdir: 'fair_trees/neighbors',
  install: true,
)
