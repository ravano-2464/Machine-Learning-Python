const form = document.getElementById("scan-form");
const fileInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const uploadPreview = document.getElementById("upload-preview");
const uploadPreviewEmpty = document.getElementById("upload-preview-empty");
const uploadPreviewImage = document.getElementById("image-upload-preview");
const uploadPreviewFileName = document.getElementById("preview-file-name");
const submitButton = document.getElementById("submit-button");
const statusBanner = document.getElementById("status-banner");
const measurementGrid = document.getElementById("measurement-grid");
const detectionsBody = document.getElementById("detections-body");
const metadataJson = document.getElementById("metadata-json");
const qualityMeterBar = document.getElementById("quality-meter-bar");
const weightsInput = form.elements.namedItem("weights");
const deviceInput = form.elements.namedItem("device");
const weightsPreset = document.getElementById("weights-preset");
const weightsPresetToggle = document.getElementById("weights-preset-toggle");
const weightsPresetLabel = document.getElementById("weights-preset-label");
const weightsPresetMenu = document.getElementById("weights-preset-menu");
const weightsPresetOptions = Array.from(weightsPreset?.querySelectorAll(".custom-select__option") || []);
const deviceSelect = document.getElementById("device-select");
const deviceSelectToggle = document.getElementById("device-select-toggle");
const deviceSelectLabel = document.getElementById("device-select-label");
const deviceSelectMenu = document.getElementById("device-select-menu");
const deviceSelectOptions = Array.from(deviceSelect?.querySelectorAll(".custom-select__option") || []);
const weightsNote = document.getElementById("weights-note");

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
const heroEls = {
    weights: document.getElementById("hero-weights"),
    imgsz: document.getElementById("hero-imgsz"),
    conf: document.getElementById("hero-conf"),
};
const customScrollbarControllers = new Map();
let apiBase = null;
let uploadPreviewUrl = null;
const defaultWeights = window.SCANNER_DEFAULT_WEIGHTS || "yolov8n.pt";
const modelPresets = Array.isArray(window.SCANNER_MODEL_PRESETS) ? window.SCANNER_MODEL_PRESETS : [];
const presetPlaceholder = weightsPreset?.dataset.placeholder || "Pilih preset model";
const devicePlaceholder = deviceSelect?.dataset.placeholder || "Pilih device";
const modelPresetLookup = new Map(
    modelPresets
        .filter((item) => item?.value)
        .map((item) => [String(item.value).trim(), item])
);

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

function findPresetByValue(value) {
    return modelPresetLookup.get((value || "").trim()) || null;
}

function getPresetOptionByValue(value) {
    const normalizedValue = (value || "").trim();
    return weightsPresetOptions.find((option) => option.dataset.value === normalizedValue) || null;
}

function clampNumber(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}

function getScrollbarSize() {
    if (window.innerWidth <= 480) {
        return 4;
    }
    if (window.innerWidth <= 768) {
        return 6;
    }
    return 8;
}

function getMinimumThumbSize() {
    if (window.innerWidth <= 480) {
        return 30;
    }
    if (window.innerWidth <= 768) {
        return 35;
    }
    return 40;
}

function refreshCustomScrollbars() {
    customScrollbarControllers.forEach((controller) => controller.update());
}

function initializeCustomScrollbar(target) {
    if (!target || customScrollbarControllers.has(target)) {
        return;
    }

    const axis = target.dataset.customScrollbar || "y";
    const hasVertical = axis === "y" || axis === "both";
    const hasHorizontal = axis === "x" || axis === "both";
    const isPageScrollArea = target.classList.contains("page-scroll-area");
    const parent = target.parentNode;
    if (!parent) {
        return;
    }

    const shell = document.createElement("div");
    shell.className = "custom-scroll-shell";
    if (hasVertical) {
        shell.classList.add("has-axis-y");
    }
    if (hasHorizontal) {
        shell.classList.add("has-axis-x");
    }
    if (isPageScrollArea) {
        shell.classList.add("custom-scroll-shell--page");
    }
    parent.insertBefore(shell, target);
    shell.appendChild(target);
    target.classList.add("custom-scroll-target");

    let verticalTrack = null;
    let verticalThumb = null;
    let horizontalTrack = null;
    let horizontalThumb = null;
    let hideTimer = null;
    let isDragging = false;
    let dragAxis = null;
    let startPointerX = 0;
    let startPointerY = 0;
    let startScrollTop = 0;
    let startScrollLeft = 0;
    let verticalThumbSize = 0;
    let horizontalThumbSize = 0;
    const handlePageScrollKeys = (event) => {
        if (!isPageScrollArea || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) {
            return;
        }

        const activeElement = document.activeElement;
        const isInteractiveElement = activeElement
            && activeElement !== document.body
            && activeElement !== document.documentElement
            && activeElement !== target
            && (
                activeElement.isContentEditable
                || Boolean(activeElement.closest("input, textarea, select, button, a[href], summary, [role='button'], [role='link'], [contenteditable='true']"))
            );
        if (isInteractiveElement) {
            return;
        }

        const currentScrollTop = target.scrollTop;
        const pageJump = Math.max(40, target.clientHeight * 0.9);
        let nextScrollTop = null;

        switch (event.key) {
            case "ArrowDown":
                nextScrollTop = currentScrollTop + 40;
                break;
            case "ArrowUp":
                nextScrollTop = currentScrollTop - 40;
                break;
            case "PageDown":
                nextScrollTop = currentScrollTop + pageJump;
                break;
            case "PageUp":
                nextScrollTop = currentScrollTop - pageJump;
                break;
            case "Home":
                nextScrollTop = 0;
                break;
            case "End":
                nextScrollTop = target.scrollHeight;
                break;
            case " ":
                nextScrollTop = currentScrollTop + (event.shiftKey ? -pageJump : pageJump);
                break;
            default:
                break;
        }

        if (nextScrollTop === null) {
            return;
        }

        event.preventDefault();
        target.scrollTop = clampNumber(nextScrollTop, 0, Math.max(0, target.scrollHeight - target.clientHeight));
        update();
        showActivity();
    };

    if (hasVertical) {
        verticalTrack = document.createElement("div");
        verticalTrack.className = "custom-scroll-track custom-scroll-track--y";
        verticalThumb = document.createElement("div");
        verticalThumb.className = "custom-scroll-thumb";
        verticalTrack.appendChild(verticalThumb);
        shell.appendChild(verticalTrack);
    }

    if (hasHorizontal) {
        horizontalTrack = document.createElement("div");
        horizontalTrack.className = "custom-scroll-track custom-scroll-track--x";
        horizontalThumb = document.createElement("div");
        horizontalThumb.className = "custom-scroll-thumb";
        horizontalTrack.appendChild(horizontalThumb);
        shell.appendChild(horizontalTrack);
    }

    const showActivity = () => {
        shell.classList.add("is-active");
        if (hideTimer) {
            window.clearTimeout(hideTimer);
        }
        if (!isDragging) {
            hideTimer = window.setTimeout(() => {
                shell.classList.remove("is-active");
            }, 900);
        }
    };

    const update = () => {
        shell.style.setProperty("--custom-scrollbar-size", `${getScrollbarSize()}px`);
        const minimumThumbSize = getMinimumThumbSize();

        if (hasVertical && verticalTrack && verticalThumb) {
            const containerHeight = target.clientHeight;
            const contentHeight = target.scrollHeight;
            const trackHeight = verticalTrack.clientHeight || containerHeight;
            const isScrollable = contentHeight - containerHeight > 1;
            shell.classList.toggle("is-scrollable-y", isScrollable);
            verticalTrack.hidden = !isScrollable;

            if (isScrollable) {
                const rawThumbHeight = (containerHeight / contentHeight) * trackHeight;
                verticalThumbSize = clampNumber(
                    rawThumbHeight,
                    Math.min(minimumThumbSize, trackHeight),
                    trackHeight
                );
                const maxThumbTop = Math.max(0, trackHeight - verticalThumbSize);
                const maxScrollTop = Math.max(0, contentHeight - containerHeight);
                const thumbTop = maxScrollTop > 0 ? (target.scrollTop / maxScrollTop) * maxThumbTop : 0;

                verticalThumb.style.height = `${verticalThumbSize}px`;
                verticalThumb.style.transform = `translateY(${clampNumber(thumbTop, 0, maxThumbTop)}px)`;
            }
        }

        if (hasHorizontal && horizontalTrack && horizontalThumb) {
            const containerWidth = target.clientWidth;
            const contentWidth = target.scrollWidth;
            const trackWidth = horizontalTrack.clientWidth || containerWidth;
            const isScrollable = contentWidth - containerWidth > 1;
            shell.classList.toggle("is-scrollable-x", isScrollable);
            horizontalTrack.hidden = !isScrollable;

            if (isScrollable) {
                const rawThumbWidth = (containerWidth / contentWidth) * trackWidth;
                horizontalThumbSize = clampNumber(
                    rawThumbWidth,
                    Math.min(minimumThumbSize, trackWidth),
                    trackWidth
                );
                const maxThumbLeft = Math.max(0, trackWidth - horizontalThumbSize);
                const maxScrollLeft = Math.max(0, contentWidth - containerWidth);
                const thumbLeft = maxScrollLeft > 0 ? (target.scrollLeft / maxScrollLeft) * maxThumbLeft : 0;

                horizontalThumb.style.width = `${horizontalThumbSize}px`;
                horizontalThumb.style.transform = `translateX(${clampNumber(thumbLeft, 0, maxThumbLeft)}px)`;
            }
        }
    };

    const stopDragging = () => {
        if (!isDragging) {
            return;
        }

        isDragging = false;
        dragAxis = null;
        shell.classList.remove("is-dragging");
        document.removeEventListener("pointermove", handlePointerMove);
        document.removeEventListener("pointerup", stopDragging);
        document.removeEventListener("pointercancel", stopDragging);
        showActivity();
    };

    function handlePointerMove(event) {
        if (!isDragging) {
            return;
        }

        if (dragAxis === "y" && hasVertical && verticalTrack) {
            const deltaY = event.clientY - startPointerY;
            const containerHeight = target.clientHeight;
            const contentHeight = target.scrollHeight;
            const trackHeight = verticalTrack.clientHeight;
            const maxThumbTop = Math.max(1, trackHeight - verticalThumbSize);
            const scrollableHeight = Math.max(0, contentHeight - containerHeight);
            const scrollDelta = (deltaY / maxThumbTop) * scrollableHeight;
            target.scrollTop = startScrollTop + scrollDelta;
        }

        if (dragAxis === "x" && hasHorizontal && horizontalTrack) {
            const deltaX = event.clientX - startPointerX;
            const containerWidth = target.clientWidth;
            const contentWidth = target.scrollWidth;
            const trackWidth = horizontalTrack.clientWidth;
            const maxThumbLeft = Math.max(1, trackWidth - horizontalThumbSize);
            const scrollableWidth = Math.max(0, contentWidth - containerWidth);
            const scrollDelta = (deltaX / maxThumbLeft) * scrollableWidth;
            target.scrollLeft = startScrollLeft + scrollDelta;
        }

        update();
    }

    const startDragging = (event, axisName) => {
        event.preventDefault();
        isDragging = true;
        dragAxis = axisName;
        startPointerX = event.clientX;
        startPointerY = event.clientY;
        startScrollTop = target.scrollTop;
        startScrollLeft = target.scrollLeft;
        shell.classList.add("is-active", "is-dragging");
        if (hideTimer) {
            window.clearTimeout(hideTimer);
        }
        document.addEventListener("pointermove", handlePointerMove);
        document.addEventListener("pointerup", stopDragging);
        document.addEventListener("pointercancel", stopDragging);
    };

    const handleTrackJump = (event, axisName) => {
        if (axisName === "y" && verticalTrack && event.target !== verticalThumb) {
            const rect = verticalTrack.getBoundingClientRect();
            const maxScrollTop = Math.max(0, target.scrollHeight - target.clientHeight);
            const maxThumbTop = Math.max(1, rect.height - verticalThumbSize);
            const clickOffset = event.clientY - rect.top - verticalThumbSize / 2;
            const scrollRatio = clampNumber(clickOffset / maxThumbTop, 0, 1);
            target.scrollTop = scrollRatio * maxScrollTop;
            update();
            showActivity();
        }

        if (axisName === "x" && horizontalTrack && event.target !== horizontalThumb) {
            const rect = horizontalTrack.getBoundingClientRect();
            const maxScrollLeft = Math.max(0, target.scrollWidth - target.clientWidth);
            const maxThumbLeft = Math.max(1, rect.width - horizontalThumbSize);
            const clickOffset = event.clientX - rect.left - horizontalThumbSize / 2;
            const scrollRatio = clampNumber(clickOffset / maxThumbLeft, 0, 1);
            target.scrollLeft = scrollRatio * maxScrollLeft;
            update();
            showActivity();
        }
    };

    target.addEventListener("scroll", () => {
        update();
        showActivity();
    }, { passive: true });

    shell.addEventListener("mouseenter", showActivity);
    shell.addEventListener("mouseleave", () => {
        if (!isDragging) {
            showActivity();
        }
    });

    verticalThumb?.addEventListener("pointerdown", (event) => startDragging(event, "y"));
    horizontalThumb?.addEventListener("pointerdown", (event) => startDragging(event, "x"));
    verticalTrack?.addEventListener("pointerdown", (event) => handleTrackJump(event, "y"));
    horizontalTrack?.addEventListener("pointerdown", (event) => handleTrackJump(event, "x"));

    const resizeObserver = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => window.requestAnimationFrame(update))
        : null;
    resizeObserver?.observe(target);
    resizeObserver?.observe(shell);

    const mutationObserver = typeof MutationObserver === "function"
        ? new MutationObserver(() => window.requestAnimationFrame(update))
        : null;
    mutationObserver?.observe(target, { childList: true, subtree: true, characterData: true });

    window.addEventListener("resize", update);
    if (isPageScrollArea) {
        window.addEventListener("keydown", handlePageScrollKeys);
    }
    window.requestAnimationFrame(update);

    customScrollbarControllers.set(target, {
        update,
        activate: showActivity,
        destroy() {
            stopDragging();
            resizeObserver?.disconnect();
            mutationObserver?.disconnect();
            window.removeEventListener("resize", update);
            if (isPageScrollArea) {
                window.removeEventListener("keydown", handlePageScrollKeys);
            }
            customScrollbarControllers.delete(target);
        },
    });
}

function initializeCustomScrollbars() {
    document.querySelectorAll("[data-custom-scrollbar]").forEach((element) => {
        initializeCustomScrollbar(element);
    });
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
        detectionsBody.innerHTML = `
            <tr>
                <td colspan="5" class="table-empty">
                    <div class="table-empty__content">Tidak ada objek yang terdeteksi.</div>
                </td>
            </tr>
        `;
        return;
    }

    detectionsBody.innerHTML = detections
        .map((item, index) => {
            const bbox = Array.isArray(item.bbox) ? item.bbox.map((value) => Number(value).toFixed(1)).join(", ") : "-";
            return `
                <tr>
                    <td class="table-cell-no">${index + 1}</td>
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

function updateUploadPreview(file) {
    if (uploadPreviewUrl) {
        URL.revokeObjectURL(uploadPreviewUrl);
        uploadPreviewUrl = null;
    }

    if (!file || !uploadPreview || !uploadPreviewImage || !uploadPreviewFileName) {
        uploadPreview?.setAttribute("hidden", "");
        uploadPreview?.classList.remove("has-image");
        if (uploadPreviewImage) {
            uploadPreviewImage.removeAttribute("src");
        }
        if (uploadPreviewEmpty) {
            uploadPreviewEmpty.hidden = false;
        }
        if (uploadPreviewFileName) {
            uploadPreviewFileName.textContent = "Belum ada file";
        }
        return;
    }

    uploadPreviewUrl = URL.createObjectURL(file);
    uploadPreview.hidden = false;
    uploadPreview.classList.remove("has-image");
    if (uploadPreviewEmpty) {
        uploadPreviewEmpty.hidden = false;
    }
    uploadPreviewImage.src = uploadPreviewUrl;
    uploadPreviewFileName.textContent = file.name;
}

uploadPreviewImage?.addEventListener("load", () => {
    uploadPreview?.classList.add("has-image");
    if (uploadPreviewEmpty) {
        uploadPreviewEmpty.hidden = true;
    }
});

uploadPreviewImage?.addEventListener("error", () => {
    uploadPreview?.classList.remove("has-image");
    uploadPreviewImage.removeAttribute("src");
    if (uploadPreviewEmpty) {
        uploadPreviewEmpty.hidden = false;
    }
});

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
    window.requestAnimationFrame(refreshCustomScrollbars);
}

function syncWeightsUi() {
    const weights = weightsInput?.value?.trim() || defaultWeights;
    const preset = findPresetByValue(weights);

    if (weightsPresetLabel) {
        weightsPresetLabel.textContent = preset
            ? `${preset.label} • ${preset.value} • ${preset.is_local ? "Lokal" : "Auto-download"}`
            : presetPlaceholder;
    }

    weightsPresetOptions.forEach((option) => {
        const isSelected = option.dataset.value === (preset?.value || "");
        option.classList.toggle("is-selected", isSelected);
        option.setAttribute("aria-selected", String(isSelected));
    });

    if (!weightsNote) {
        return;
    }

    if (preset) {
        weightsNote.textContent = `${preset.label} • ${preset.description} ${preset.availability}. Weights dikunci mengikuti model ini.`;
        return;
    }

    weightsNote.textContent = "Weights dikunci dan mengikuti preset model yang dipilih.";
}

function getDeviceOptionByValue(value) {
    const normalizedValue = (value || "").trim();
    return deviceSelectOptions.find((option) => option.dataset.value === normalizedValue) || null;
}

function syncDeviceUi() {
    const device = deviceInput?.value?.trim() || "cpu";
    const selectedOption = getDeviceOptionByValue(device);

    if (deviceSelectLabel) {
        deviceSelectLabel.textContent = selectedOption
            ? `${selectedOption.dataset.label || device.toUpperCase()}`
            : devicePlaceholder;
    }

    deviceSelectOptions.forEach((option) => {
        const isSelected = option.dataset.value === device;
        option.classList.toggle("is-selected", isSelected);
        option.setAttribute("aria-selected", String(isSelected));
    });
}

function closeWeightsPresetMenu({ restoreFocus = false } = {}) {
    if (!weightsPreset) {
        return;
    }
    weightsPreset.classList.remove("is-open");
    weightsPresetToggle?.setAttribute("aria-expanded", "false");
    if (restoreFocus) {
        weightsPresetToggle?.focus();
    }
}

function openWeightsPresetMenu() {
    if (!weightsPreset) {
        return;
    }
    closeDeviceSelectMenu();
    weightsPreset.classList.add("is-open");
    weightsPresetToggle?.setAttribute("aria-expanded", "true");
    window.requestAnimationFrame(() => {
        refreshCustomScrollbars();
        const menuScroller = weightsPresetMenu?.querySelector("[data-custom-scrollbar]");
        customScrollbarControllers.get(menuScroller)?.activate();
    });
    const selectedOption = getPresetOptionByValue(weightsInput?.value) || weightsPresetOptions[0];
    window.requestAnimationFrame(() => {
        selectedOption?.focus();
    });
}

function closeDeviceSelectMenu({ restoreFocus = false } = {}) {
    if (!deviceSelect) {
        return;
    }
    deviceSelect.classList.remove("is-open");
    deviceSelectToggle?.setAttribute("aria-expanded", "false");
    if (restoreFocus) {
        deviceSelectToggle?.focus();
    }
}

function openDeviceSelectMenu() {
    if (!deviceSelect) {
        return;
    }
    closeWeightsPresetMenu();
    deviceSelect.classList.add("is-open");
    deviceSelectToggle?.setAttribute("aria-expanded", "true");
    window.requestAnimationFrame(() => {
        refreshCustomScrollbars();
        const menuScroller = deviceSelectMenu?.querySelector("[data-custom-scrollbar]");
        customScrollbarControllers.get(menuScroller)?.activate();
    });
    const selectedOption = getDeviceOptionByValue(deviceInput?.value) || deviceSelectOptions[0];
    window.requestAnimationFrame(() => {
        selectedOption?.focus();
    });
}

function focusWeightsPresetOption(direction) {
    if (!weightsPresetOptions.length) {
        return;
    }
    const activeIndex = weightsPresetOptions.findIndex((option) => option === document.activeElement);
    if (activeIndex === -1) {
        (getPresetOptionByValue(weightsInput?.value) || weightsPresetOptions[0])?.focus();
        return;
    }
    const nextIndex = (activeIndex + direction + weightsPresetOptions.length) % weightsPresetOptions.length;
    weightsPresetOptions[nextIndex]?.focus();
}

function focusDeviceOption(direction) {
    if (!deviceSelectOptions.length) {
        return;
    }
    const activeIndex = deviceSelectOptions.findIndex((option) => option === document.activeElement);
    if (activeIndex === -1) {
        (getDeviceOptionByValue(deviceInput?.value) || deviceSelectOptions[0])?.focus();
        return;
    }
    const nextIndex = (activeIndex + direction + deviceSelectOptions.length) % deviceSelectOptions.length;
    deviceSelectOptions[nextIndex]?.focus();
}

function buildBackendCandidates() {
    const candidates = [];
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
        candidates.push(window.location.origin);
    }
    candidates.push("http://127.0.0.1:5000", "http://localhost:5000");
    return [...new Set(candidates)];
}

async function readJsonResponse(response) {
    const raw = await response.text();
    if (!raw.trim()) {
        throw new Error("Respons dari backend kosong. Pastikan server Flask berjalan di http://127.0.0.1:5000.");
    }

    try {
        return JSON.parse(raw);
    } catch {
        if (!response.ok) {
            throw new Error(`Backend mengembalikan respons non-JSON (status ${response.status}). Pastikan frontend terhubung ke Flask app.`);
        }
        throw new Error("Backend mengembalikan format yang tidak bisa dibaca. Pastikan endpoint /scan berasal dari Flask app ini.");
    }
}

async function detectBackend(showStatus = false) {
    const candidates = buildBackendCandidates();

    for (const candidate of candidates) {
        try {
            const response = await fetch(`${candidate}/health`, {
                method: "GET",
                mode: "cors",
                cache: "no-store",
            });
            const data = await readJsonResponse(response);
            if (response.ok && data?.status === "ok") {
                apiBase = candidate;
                if (showStatus) {
                    const suffix = candidate === window.location.origin ? "Backend aktif dan siap dipakai." : `Backend aktif di ${candidate}.`;
                    setStatus(suffix, "success");
                }
                return candidate;
            }
        } catch {
            continue;
        }
    }

    apiBase = null;
    if (showStatus) {
        setStatus("Backend tidak terdeteksi. Jalankan `python -m webapp.app` lalu buka http://127.0.0.1:5000.", "error");
    }
    return null;
}

function syncHeroStats() {
    const weights = weightsInput?.value || defaultWeights;
    const imgsz = form.elements.namedItem("imgsz")?.value || "640";
    const conf = form.elements.namedItem("conf")?.value || "0.25";

    heroEls.weights.textContent = weights;
    heroEls.imgsz.textContent = `${imgsz} px`;
    heroEls.conf.textContent = conf;
}

function getProcessingStatusMessage() {
    const preset = findPresetByValue(weightsInput?.value);
    if (preset && !preset.is_local) {
        return `Model ${preset.value} akan dipersiapkan dulu. Pemakaian pertama bisa sedikit lebih lama karena weights diunduh otomatis.`;
    }
    return "Model sedang memproses gambar, tunggu sebentar...";
}

fileInput.addEventListener("change", () => {
    const selectedFile = fileInput.files?.[0];
    updateFileLabel(selectedFile);
    updateUploadPreview(selectedFile);
});

["imgsz", "conf"].forEach((fieldName) => {
    const element = form.elements.namedItem(fieldName);
    element?.addEventListener("input", syncHeroStats);
});

weightsPresetToggle?.addEventListener("click", () => {
    if (weightsPreset?.classList.contains("is-open")) {
        closeWeightsPresetMenu();
        return;
    }
    openWeightsPresetMenu();
});

weightsPresetToggle?.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openWeightsPresetMenu();
    }
    if (event.key === "Escape") {
        closeWeightsPresetMenu();
    }
});

weightsPresetMenu?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        event.preventDefault();
        closeWeightsPresetMenu({ restoreFocus: true });
        return;
    }
    if (event.key === "ArrowDown") {
        event.preventDefault();
        focusWeightsPresetOption(1);
        return;
    }
    if (event.key === "ArrowUp") {
        event.preventDefault();
        focusWeightsPresetOption(-1);
    }
});

weightsPresetOptions.forEach((option) => {
    option.addEventListener("click", () => {
        if (weightsInput) {
            weightsInput.value = option.dataset.value || "";
        }
        syncHeroStats();
        syncWeightsUi();
        closeWeightsPresetMenu({ restoreFocus: true });
    });
});

deviceSelectToggle?.addEventListener("click", () => {
    if (deviceSelect?.classList.contains("is-open")) {
        closeDeviceSelectMenu();
        return;
    }
    openDeviceSelectMenu();
});

deviceSelectToggle?.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        openDeviceSelectMenu();
    }
    if (event.key === "Escape") {
        closeDeviceSelectMenu();
    }
});

deviceSelectMenu?.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        event.preventDefault();
        closeDeviceSelectMenu({ restoreFocus: true });
        return;
    }
    if (event.key === "ArrowDown") {
        event.preventDefault();
        focusDeviceOption(1);
        return;
    }
    if (event.key === "ArrowUp") {
        event.preventDefault();
        focusDeviceOption(-1);
    }
});

deviceSelectOptions.forEach((option) => {
    option.addEventListener("click", () => {
        if (deviceInput) {
            deviceInput.value = option.dataset.value || "cpu";
        }
        syncDeviceUi();
        closeDeviceSelectMenu({ restoreFocus: true });
    });
});

document.addEventListener("click", (event) => {
    if (!weightsPreset?.contains(event.target)) {
        closeWeightsPresetMenu();
    }
    if (!deviceSelect?.contains(event.target)) {
        closeDeviceSelectMenu();
    }
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
    updateUploadPreview(droppedFile);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!fileInput.files?.length) {
        setStatus("Pilih file gambar terlebih dahulu.", "error");
        return;
    }

    if (!apiBase) {
        await detectBackend(false);
    }
    if (!apiBase) {
        setStatus("Backend tidak terdeteksi. Jalankan `python -m webapp.app` lalu buka http://127.0.0.1:5000.", "error");
        return;
    }

    submitButton.disabled = true;
    setStatus(getProcessingStatusMessage(), "");

    try {
        const payload = new FormData(form);
        const response = await fetch(`${apiBase}/scan`, {
            method: "POST",
            mode: "cors",
            body: payload,
        });
        const data = await readJsonResponse(response);

        if (!response.ok) {
            throw new Error(data.error || "Terjadi kesalahan saat memproses gambar.");
        }

        renderSummary(data.summary || {});
        renderMeasurements(data.measurements || {});
        renderDetections(data.detections || []);
        renderImages(data.images || {});
        metadataJson.textContent = JSON.stringify(data.scan || data, null, 2);
        window.requestAnimationFrame(refreshCustomScrollbars);
        setStatus(data.summary?.status || "Scan selesai.", data.summary?.target_label ? "success" : "");
    } catch (error) {
        resetResults();
        setStatus(error.message || "Terjadi kesalahan tak terduga.", "error");
    } finally {
        submitButton.disabled = false;
    }
});

initializeCustomScrollbars();
resetResults();
updateUploadPreview(fileInput.files?.[0]);
syncHeroStats();
syncWeightsUi();
syncDeviceUi();
detectBackend(true);
