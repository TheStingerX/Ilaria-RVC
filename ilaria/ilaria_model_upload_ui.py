"""Binary model-import controls for the bundled Gradio interface."""

_UPLOAD_PANEL = """
<style>
    #ilaria-binary-model-import {
        display: grid;
        gap: .7rem;
        padding: .25rem;
    }
    #ilaria-binary-model-import [hidden] { display: none !important; }
    #ilaria-binary-model-import .ilaria-model-drop-zone {
        display: grid;
        gap: .55rem;
        justify-items: center;
        border: 2px dashed #d8b4fe;
        border-radius: 12px;
        background: #4f3557;
        padding: 1rem;
        color: #fff4fc;
        text-align: center;
    }
    #ilaria-binary-model-import .ilaria-model-drop-zone[data-dragging] {
        border-color: #ffc0cb;
        background: #674367;
    }
    #ilaria-binary-model-import .ilaria-model-actions {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
    }
    #ilaria-binary-model-import button {
        min-height: 2.4rem;
        padding: .4rem .85rem;
        border-radius: 8px;
        background: linear-gradient(115deg, #ffc0cb, #e6b7f6);
        color: #2d1833;
        font-weight: 600;
    }
    #ilaria-binary-model-import button:disabled { opacity: .5; cursor: not-allowed; }
    #ilaria-binary-model-import button:focus-visible {
        outline: 2px solid #f9e5ff;
        outline-offset: 2px;
    }
    #ilaria-binary-model-import [data-selected] {
        display: grid;
        gap: .2rem;
        max-height: 9rem;
        overflow: auto;
        overflow-wrap: anywhere;
        color: #f8e9fa;
        margin: 0;
    }
    #ilaria-binary-model-import progress {
        width: 100%;
        height: .65rem;
        accent-color: #ffc0cb;
    }
    #ilaria-binary-model-import [data-status] {
        color: #fff4fc;
        overflow-wrap: anywhere;
        min-height: 1.4rem;
    }
</style>
<div id="ilaria-binary-model-import">
    <div class="ilaria-model-drop-zone" data-drop-zone>
        <span>Drop .pth and .index files or a .zip here</span>
        <input data-file-input type="file" accept=".zip,.pth,.index" multiple hidden>
        <button data-choose type="button">Choose files</button>
    </div>
    <ul data-selected aria-label="Selected model files"></ul>
    <div class="ilaria-model-actions">
        <button data-import type="button" disabled>Import model files</button>
        <button data-cancel type="button" hidden>Cancel import</button>
    </div>
    <progress data-progress value="0" max="100" aria-label="Model import progress" hidden></progress>
    <div data-status role="status" aria-live="polite">Select files to import.</div>
</div>
"""


def build_binary_model_upload(gr):
    """Render the file controls. The app load hook attaches the upload code."""
    gr.HTML(_UPLOAD_PANEL, elem_id="binary-model-upload")


BINARY_MODEL_UPLOAD_JS = r"""
() => {
    const host = document.querySelector("gradio-app");
    const root = host?.shadowRoot || host || document;
    const maxFiles = 64;
    const maxBytes = 16 * 1024 ** 3;
    const maxHeader = 12000;

    function mount() {
        const panel = root.querySelector("#ilaria-binary-model-import");
        if (!panel) return false;
        if (panel.dataset.ready === "true") return true;
        panel.dataset.ready = "true";

        const zone = panel.querySelector("[data-drop-zone]");
        const input = panel.querySelector("[data-file-input]");
        const choose = panel.querySelector("[data-choose]");
        const selected = panel.querySelector("[data-selected]");
        const start = panel.querySelector("[data-import]");
        const cancel = panel.querySelector("[data-cancel]");
        const bar = panel.querySelector("[data-progress]");
        const status = panel.querySelector("[data-status]");
        let files = [];
        let manifest = "";
        let active = null;

        const sizeText = bytes => bytes >= 1024 ** 3
            ? (bytes / 1024 ** 3).toFixed(2) + " GiB"
            : (bytes / 1024 ** 2).toFixed(1) + " MiB";

        function setStatus(message, percent) {
            status.textContent = message;
            if (typeof percent === "number" && Number.isFinite(percent)) {
                bar.hidden = false;
                bar.value = Math.min(100, Math.max(0, percent));
            }
        }

        function setBusy(busy) {
            choose.disabled = input.disabled = busy;
            start.disabled = busy || !files.length;
            cancel.hidden = !busy;
            panel.setAttribute("aria-busy", String(busy));
        }

        function select(candidate) {
            if (active) return;
            files = [];
            manifest = "";
            selected.replaceChildren();
            bar.value = 0;
            bar.hidden = true;
            const next = Array.from(candidate);
            const total = next.reduce((sum, file) => sum + file.size, 0);
            if (!next.length || next.length > maxFiles) {
                setStatus("Select between 1 and 64 files.");
            } else if (next.some(file => !/\.(zip|pth|index)$/i.test(file.name) || file.size <= 0)) {
                setStatus("Choose nonempty .zip, .pth, or .index files.");
            } else if (total > maxBytes) {
                setStatus("The selected files exceed the 16 GiB upload limit.");
            } else {
                manifest = encodeURIComponent(JSON.stringify(next.map(file => ({name: file.name, size: file.size}))));
                if (manifest.length > maxHeader) {
                    setStatus("The file names are too long for one upload. Select fewer files and try again.");
                } else {
                    files = next;
                    for (const file of files) {
                        const line = document.createElement("li");
                        line.textContent = file.name + " · " + sizeText(file.size);
                        selected.append(line);
                    }
                    setStatus(files.length + " file(s) ready · " + sizeText(total));
                }
            }
            setBusy(false);
        }

        choose.addEventListener("click", () => input.click());
        input.addEventListener("change", () => {
            select(input.files);
            input.value = "";
        });
        zone.addEventListener("dragover", event => {
            if (!event.dataTransfer?.types.includes("Files")) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = active ? "none" : "copy";
            if (!active) zone.setAttribute("data-dragging", "");
        });
        zone.addEventListener("dragleave", event => {
            if (!zone.contains(event.relatedTarget)) zone.removeAttribute("data-dragging");
        });
        zone.addEventListener("drop", event => {
            event.preventDefault();
            zone.removeAttribute("data-dragging");
            if (event.dataTransfer) select(event.dataTransfer.files);
        });
        cancel.addEventListener("click", () => {
            if (active) active.abort();
        });

        start.addEventListener("click", () => {
            if (active || !files.length) return;
            const xhr = new XMLHttpRequest();
            active = xhr;
            setBusy(true);
            setStatus("Uploading files…", 0);
            let responseOffset = 0;
            let pending = "";
            let lastEvent = null;

            function handleLine(line) {
                if (!line.trim()) return;
                try {
                    const event = JSON.parse(line);
                    if (typeof event.message !== "string") return;
                    lastEvent = event;
                    const progress = Number.isFinite(event.percent)
                        ? 50 + Math.min(100, Math.max(0, event.percent)) / 2 : undefined;
                    setStatus(event.message, progress);
                } catch (_) {
                    // Ignore an invalid progress line without losing the rest of the response.
                }
            }

            function readProgress(flush = false) {
                pending += xhr.responseText.slice(responseOffset);
                responseOffset = xhr.responseText.length;
                const lines = pending.split("\n");
                pending = lines.pop();
                for (const line of lines) handleLine(line);
                if (flush) handleLine(pending);
            }

            function finish(message, succeeded = false) {
                if (active !== xhr) return;
                active = null;
                if (succeeded) {
                    files = [];
                    manifest = "";
                    selected.replaceChildren();
                }
                setBusy(false);
                if (!succeeded) bar.hidden = true;
                setStatus(message, succeeded ? 100 : undefined);
            }

            xhr.upload.addEventListener("progress", event => {
                if (lastEvent) return;
                if (event.lengthComputable) {
                    const percent = 100 * event.loaded / event.total;
                    setStatus("Uploading " + sizeText(event.loaded) + " / " + sizeText(event.total)
                              + " (" + percent.toFixed(0) + "%)…", percent / 2);
                } else {
                    setStatus("Uploading " + sizeText(event.loaded) + "…");
                }
            });
            xhr.upload.addEventListener("load", () => {
                if (!lastEvent) setStatus("Upload complete. Checking model files…", 50);
            });
            xhr.addEventListener("progress", () => {
                if (xhr.status === 200) readProgress();
            });
            xhr.addEventListener("load", () => {
                if (xhr.status !== 200) {
                    let message = "Upload failed (HTTP " + xhr.status + ").";
                    try {
                        const body = JSON.parse(xhr.responseText);
                        if (typeof body.detail === "string") message = body.detail;
                        else if (typeof body.message === "string") message = body.message;
                    } catch (_) {}
                    finish(message);
                    return;
                }
                readProgress(true);
                const succeeded = lastEvent?.done === true && !lastEvent.error;
                finish(lastEvent?.message || "The import ended without a result. Try again.", succeeded);
            });
            xhr.addEventListener("error", () => finish("Could not reach Ilaria RVC. Try again."));
            xhr.addEventListener("abort", () => finish("Import canceled."));
            xhr.addEventListener("timeout", () => finish("The import timed out. Try again."));
            try {
                xhr.open("POST", "ilaria/model-import");
                xhr.setRequestHeader("Content-Type", "application/octet-stream");
                xhr.setRequestHeader("X-Ilaria-Files", manifest);
                xhr.send(new Blob(files, {type: "application/octet-stream"}));
            } catch (_) {
                finish("Could not start the upload. Try again.");
            }
        });
        return true;
    }

    if (!mount()) {
        const observer = new MutationObserver(() => {
            if (mount()) observer.disconnect();
        });
        observer.observe(root, {childList: true, subtree: true});
        window.setTimeout(() => observer.disconnect(), 30000);
    }
    return [];
}
"""
