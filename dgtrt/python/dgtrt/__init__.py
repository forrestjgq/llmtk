import os
import glob

DIR = os.path.abspath(os.path.dirname(__file__))

def get_include_dir() -> str:
    installed_path = os.path.join(DIR, "include")
    source_path = os.path.join(os.path.dirname(DIR), "include")
    return installed_path if os.path.exists(installed_path) else source_path


def get_binding_so_path() -> str:

    paths = glob.glob(os.path.join(DIR, "bindings.cpython*.so"))
    if len(paths) == 1:
        return paths[0]

    msg = "dgtrt cpython so not found"
    raise ImportError(msg)

def get_cmake_linking() -> str:

    paths = glob.glob(os.path.join(DIR, "bindings.cpython*.so"))
    if len(paths) == 1:
        return f'-L{os.path.dirname(paths[0])} -lbindings'

    msg = "dgtrt cpython so not found"
    raise ImportError(msg)

def make_so_links():
    so = get_binding_so_path()
    parent = os.path.dirname(so)
    binding = os.path.join(parent, "libbindings.so")
    if not os.path.exists(binding):
        os.symlink(so, binding)

from dgtrt.bindings import *

make_so_links()
