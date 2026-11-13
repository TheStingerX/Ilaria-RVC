"""Training Dashboard controls that refresh without Gradio's event queue."""

TRAINING_METRICS_PANEL = """
<style>
#ilaria-training-metrics { display:grid; gap:1rem; padding:clamp(12px,2vw,24px); min-width:0; }
#ilaria-training-metrics > p { margin:0; padding:.35rem .65rem; line-height:1.6; color:#f8effa; }
#ilaria-training-metrics button {
    width:100%; min-height:2.6rem; padding:.65rem 1rem; border-radius:10px;
    background:linear-gradient(115deg,#ffc0cb,#d8b4fe); color:#2d1833; font-weight:600;
}
#ilaria-training-metrics button:disabled { opacity:.7; cursor:wait; }
#ilaria-training-metrics [data-status] {
    border:1px solid #d8b4fe; border-radius:10px; padding:.8rem 1rem;
    color:#fff4fc; background:#62416b;
}
#ilaria-training-metrics [hidden] { display:none !important; }
#ilaria-training-metrics [data-table] { overflow-x:auto; border:1px solid #d8b4fe; border-radius:12px; background:#62416b; }
#ilaria-training-metrics table { width:100%; border-collapse:collapse; text-align:left; }
#ilaria-training-metrics th, #ilaria-training-metrics td { padding:.55rem .8rem; border-bottom:1px solid #866190; color:#fff4fc; font-variant-numeric:tabular-nums; }
#ilaria-training-metrics th { background:#412849; color:#ffc0cb; }
#ilaria-training-metrics tbody tr:nth-child(even) { background:#55365e; }
#ilaria-training-metrics button:focus-visible { outline:2px solid #fff4fc; outline-offset:3px; }
#ilaria-training-metrics img { max-width:100%; height:auto; border-radius:10px; }
</style>
<section id="ilaria-training-metrics">
    <button type="button">Refresh training metrics</button>
    <p>Uses the Model Name entered in Training. You can switch tabs while metrics load.</p>
    <p data-status role="status" aria-live="polite"><strong>Dashboard status</strong><br>
        <span>Ready. Click Refresh training metrics to load the selected model.</span>
    </p>
    <img alt="Epoch timing and losses" hidden>
    <div data-table hidden>
        <table>
            <thead><tr><th>Epoch</th><th>Total steps</th><th>Seconds</th><th>Steps/sec</th></tr></thead>
            <tbody></tbody>
        </table>
    </div>
</section>
"""

LIVE_TRAINING_PANEL = """
<style>
#ilaria-live-training { padding: .9rem 1rem; border:1px solid #d8b4fe; border-radius:10px; }
#ilaria-live-training dl { display:grid; grid-template-columns:repeat(auto-fit,minmax(13rem,1fr)); gap:.65rem; margin:.65rem 0 0; }
#ilaria-live-training dl:empty { display:none; }
#ilaria-live-training dl > div { padding:.6rem .75rem; background:#412849; border-radius:8px; overflow-wrap:anywhere; }
#ilaria-live-training dt { color:#ffc0cb; }
#ilaria-live-training dd { margin:.2rem 0 0; color:#fff4fc; font-variant-numeric:tabular-nums; }
</style>
<section id="ilaria-live-training" aria-label="Live training metrics">
    <strong>Live training metrics</strong>
    <p data-live-status role="status">Checking for an active training run…</p>
    <dl></dl>
    <img alt="Training loss and epoch charts" style="max-width:100%;height:auto" hidden>
</section>
"""

TRAINING_METRICS_JS = r"""
() => {
    const host = document.querySelector("gradio-app");
    const root = host?.shadowRoot || host || document;
    const selectedModel = () => root.querySelector("#training-model-name textarea, #training-model-name input")?.value.trim() || "";
    let previousLiveData = "";
    let previousLivePanel;
    let chartKey = "";
    let nextChart = 0;
    let chartLoading = false;
    async function updateLiveChart(panel, model) {
        if (!model || chartLoading || Date.now() < nextChart) return;
        chartLoading = true;
        nextChart = Date.now() + 15000;
        try {
            const response = await fetch("ilaria/training-metrics?" + new URLSearchParams({model}), {
                method: "POST", cache: "no-store", signal: AbortSignal.timeout(15000)
            });
            const data = await response.json();
            const image = panel.querySelector("img");
            if (chartKey === model && data.image) {
                image.src = "data:image/png;base64," + data.image;
                image.hidden = false;
            }
        } catch (error) {
            // Keep the last chart while the metrics reader reconnects.
        } finally {
            chartLoading = false;
        }
    }
    async function pollLiveTraining() {
        const panel = root.querySelector("#ilaria-live-training");
        try {
            if (!panel || document.hidden || !panel.getClientRects().length) return;
            const status = panel.querySelector("[data-live-status]");
            const response = await fetch("ilaria/live-training?" + new URLSearchParams({model: selectedModel()}), {
                method: "POST", cache: "no-store", signal: AbortSignal.timeout(5000)
            });
            const data = await response.json();
            const model = data.model || "";
            if (panel !== previousLivePanel || chartKey !== model) {
                chartKey = model;
                nextChart = 0;
                panel.querySelector("img").hidden = true;
            }
            if (data.values?.length) void updateLiveChart(panel, model);
            const serialized = JSON.stringify(data);
            if (panel === previousLivePanel && serialized === previousLiveData) return;
            previousLivePanel = panel;
            previousLiveData = serialized;
            status.textContent = data.status || (data.active
                ? data.model + (data.values.length ? " is training." : " is training. Waiting for the first recorded values…")
                : "Last recorded values for " + data.model + (data.values.length ? "." : " are unavailable."));
            const values = panel.querySelector("dl");
            values.replaceChildren();
            for (const [label, value] of data.values || []) {
                const item = document.createElement("div");
                const term = document.createElement("dt");
                const detail = document.createElement("dd");
                term.textContent = label;
                detail.textContent = value;
                item.append(term, detail);
                values.append(item);
            }
        } catch (error) {
            previousLiveData = "";
            if (panel) panel.querySelector("[data-live-status]").textContent = "Waiting to reconnect to training metrics…";
        } finally {
            setTimeout(pollLiveTraining, 2000);
        }
    }
    pollLiveTraining();
    root.addEventListener("click", async (event) => {
        const button = event.target.closest("#ilaria-training-metrics button");
        if (!button || button.disabled) return;
        const panel = button.closest("#ilaria-training-metrics");
        const status = panel.querySelector("[data-status] span");
        const model = selectedModel();
        if (!model) {
            status.textContent = "Enter a Model Name in Training, then refresh.";
            return;
        }
        button.disabled = true;
        status.textContent = "Loading training metrics for " + model + "…";
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000);
        try {
            const params = new URLSearchParams({model});
            const response = await fetch("ilaria/training-metrics?" + params, {
                method: "POST", signal: controller.signal, cache: "no-store"
            });
            const data = await response.json();
            status.textContent = data.status || "Could not load training metrics.";
            if (!response.ok || !Array.isArray(data.rows)) return;
            const graph = panel.querySelector("img");
            if (data.image) graph.src = "data:image/png;base64," + data.image;
            else graph.removeAttribute("src");
            graph.hidden = !data.image;
            const tbody = panel.querySelector("tbody");
            tbody.replaceChildren();
            for (const values of data.rows) {
                const row = tbody.insertRow();
                for (const value of values) row.insertCell().textContent = value ?? "—";
            }
            panel.querySelector("[data-table]").hidden = !data.rows.length;
        } catch (error) {
            status.textContent = controller.signal.aborted
                ? "Reading training metrics timed out. Try refreshing again."
                : "Could not reach Ilaria RVC. Check that the program is running.";
        } finally {
            clearTimeout(timeout);
            button.disabled = false;
        }
    });
    return [];
}
"""
