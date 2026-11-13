"""Start TensorBoard with the embedded Python 3.9 stdlib ZIP loader."""

import zipimport

if not hasattr(zipimport.zipimporter, "exec_module"):
    def exec_module(loader, module):
        exec(loader.get_code(module.__name__), module.__dict__)

    zipimport.zipimporter.create_module = lambda loader, spec: None
    zipimport.zipimporter.exec_module = exec_module

from tensorboard.main import run_main

if __name__ == "__main__":
    run_main()
