const form = document.getElementById("scan-form");
const fileInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const submitButton = document.getElementById("submit-button");
const statusBanner = document.getElementById("status-banner");
const measurementGrid = document.getElementById("measurement-grid");
const detectionsBody = document.getElementById("detections-body");
const metadataJson = document.getElementById("metadata-json");
const qualityMeterBar = document.getElementById("quality-meter-bar");

const summaryEls = {
    status: document.getElementById("summary-status"),
    detections: document.getElementById("summary-detections"),
    target: document.getElementById("summary-target"),
    quality: document.getElementById("summary-quality"),
};

const imageEls = {
    original: document.getElementById("image-original"),
    annotated: document.getElementById("image-annotated"),
    crop: document.getElementById("image-crop"),
    flattened: document.getElementById("image-flattened"),
    enhanced: document.getElementById("image-enhanced"),
};

function setStatus(message, tone = "") {
    statusBanner.textContent = message;
    statusBanner.className = "status-banner";
    if (tone) {
        statusBanner.classList.add(`is-${tone}`);
    }
}

function formatValue(value) {
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : value.toFixed(4);
    }
    if (value === null || value === undefined || value === "") {
        return "-";
    }
    return String(value);
}

function renderMeasurements(measurements) {
    const entries = Object.entries(measurements || {});
    if (!entries.length) {
        measurementGrid.innerHTML = '<div class="measurement-empty">Measurement belum tersedia untuk hasil ini.</div>';
        return;
    }

    measurementGrid.innerHTML = entries
        .map(([key, value]) => {
            const label = key.replaceAll("_", " ");
            return `
                <article class="measurement-card">
                    <span>${label}</span>
                    <strong>${formatValue(value)}</strong>
                </article>
            `;
        })
        .join("");
}

function renderDetections(detections) {
    if (!detections || !detections.length) {
        detectionsBody.innerHTML = '<tr><td colspan="4" class="table-empty">Tidak ada objek yang terdeteksi.</td></tr>';
        return;
    }

    detectionsBody.innerHTML = detections
        .map((item) => {
            const bbox = Array.isArray(item.bbox) ? item.bbox.map((value) => Number(value).toFixed(1)).join(", ") : "-";
            return `
                <tr>
                    <td>${item.label ?? "-"}</td>
                    <td>${formatValue(item.conf)}</td>
                    <td>${formatValue(item.class)}</td>
                    <td>${bbox}</td>
                </tr>
            `;
        })
        .join("");
}

function renderImages(images) {
    Object.entries(imageEls).forEach(([key, element]) => {
        const src = images?.[key];
        if (src) {
            element.src = src;
            element.style.display = "block";
        } else {
            element.removeAttribute("src");
            element.style.display = "none";
        }
    });
}

function renderSummary(summary) {
    summaryEls.status.textContent = summary.status || "-";
    summaryEls.detections.textContent = formatValue(summary.detections_count);
    summaryEls.target.textContent = summary.target_label ? `${summary.target_label} (${formatValue(summary.target_confidence)})` : "-";
    summaryEls.quality.textContent = summary.quality !== null && summary.quality !== undefined ? formatValue(summary.quality) : "-";

    const quality = Number(summary.quality || 0);
    qualityMeterBar.style.width = `${Math.max(0, Math.min(100, quality * 100))}%`;
}

function updateFileLabel(file) {
    fileLabel.textContent = file ? `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB` : "Belum ada file yang dipilih";
}

function resetResults() {
    renderSummary({
        status: "Belum ada proses",
        detections_count: 0,
        target_label: null,
        target_confidence: null,
        quality: null,
    });
    renderMeasurements({});
    renderDetections([]);
    renderImages({});
    metadataJson.textContent = JSON.stringify({ message: "Hasil scan akan muncul di sini." }, null, 2);
}

fileInput.addEventListener("change", () => {
    updateFileLabel(fileInput.files?.[0]);
});

const uploadZone = document.querySelector(".upload-zone");
["dragenter", "dragover"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.add("is-dragover");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.remove("is-dragover");
    });
});

uploadZone.addEventListener("drop", (event) => {
    const droppedFile = event.dataTransfer?.files?.[0];
    if (!droppedFile) {
        return;
    }
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(droppedFile);
    fileInput.files = dataTransfer.files;
    updateFileLabel(droppedFile);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!fileInput.files?.length) {
        setStatus("Pilih file gambar terlebih dahulu.", "error");
        return;
    }

    submitButton.disabled = true;
    setStatus("Model sedang memproses gambar, tunggu sebentar...", "");

    try {
        const payload = new FormData(form);
        const response = await fetch("/scan", {
            method: "POST",
            body: payload,
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Terjadi kesalahan saat memproses gambar.");
        }

        renderSummary(data.summary || {});
        renderMeasurements(data.measurements || {});
        renderDetections(data.detections || []);
        renderImages(data.images || {});
        metadataJson.textContent = JSON.stringify(data.scan || data, null, 2);
        setStatus(data.summary?.status || "Scan selesai.", data.summary?.target_label ? "success" : "");
    } catch (error) {
        resetResults();
        setStatus(error.message || "Terjadi kesalahan tak terduga.", "error");
    } finally {
        submitButton.disabled = false;
    }
});

resetResults();
