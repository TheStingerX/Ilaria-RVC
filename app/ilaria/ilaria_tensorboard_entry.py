"""Start TensorBoard with the embedded Python 3.9 stdlib ZIP loader."""

import zipimport

if not hasattr(zipimport.zipimporter, "exec_module"):
    def exec_module(loader, module):
        exec(loader.get_code(module.__name__), module.__dict__)

    zipimport.zipimporter.create_module = lambda loader, spec: None
    zipimport.zipimporter.exec_module = exec_module

from io import BytesIO
from zipfile import ZipFile

from absl import app
from tensorboard import assets, default, main_lib, program


def live_assets():
    provider = assets.get_default_assets_zip_provider()
    output = BytesIO()
    with provider() as source, ZipFile(source) as original, ZipFile(output, "w") as patched:
        for entry in original.infolist():
            content = original.read(entry)
            if entry.filename == "index.js":
                content = content.replace(
                    b"reloadPeriodInMs:3e4,reloadEnabled:!1",
                    b"reloadPeriodInMs:3e4,reloadEnabled:!0",
                    1,
                )
            patched.writestr(entry, content)
    output.seek(0)
    return output


if __name__ == "__main__":
    main_lib.global_init()
    board = program.TensorBoard(plugins=default.get_plugins(), assets_zip_provider=live_assets)
    app.run(board.main, flags_parser=board.configure)
