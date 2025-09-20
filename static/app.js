// Advanced feature set
const GROUP_METHODS = {
  blur: [
    { value: "mean", label: "Mean (Box Blur)" },
    { value: "gaussian", label: "Gaussian" },
    { value: "median", label: "Median" },
    { value: "bilateral", label: "Bilateral (edge-preserving)" },
  ],
  sharpen: [
    { value: "unsharp", label: "Unsharp Mask" },
    { value: "laplacian_sharpen", label: "Laplacian Sharpen" },
  ],
  edge: [
    { value: "sobel", label: "Sobel" },
    { value: "prewitt", label: "Prewitt" },
    { value: "laplacian", label: "Laplacian" },
    { value: "canny", label: "Canny" },
    { value: "binary", label: "Binary Threshold" },
  ],
};

const form = document.getElementById("editForm");
const groupSel = document.getElementById("group");
const methodSel = document.getElementById("method");
const presetsSel = document.getElementById("presets");
const imageInput = document.getElementById("image");
const originalPath = document.getElementById("original_path");
const procBtn = document.getElementById("processBtn");
const downloadLink = document.getElementById("downloadLink");
const originalPreview = document.getElementById("originalPreview");
const processedPreview = document.getElementById("processedPreview");
const tabs = document.querySelectorAll(".tab");
const panelBasic = document.getElementById("panel-basic");
const panelAdvanced = document.getElementById("panel-advanced");
const advMethodSel = document.getElementById("adv_method");
const advGroupHidden = document.getElementById("adv_group");
const showHist = document.getElementById("show_hist");
const histArea = document.getElementById("hist-area");
const histBefore = document.getElementById("histBefore");
const histAfter = document.getElementById("histAfter");

const paramFields = {
  sigma: document.querySelector('[data-param="sigma"]'),
  amount: document.querySelector('[data-param="amount"]'),
  alpha: document.querySelector('[data-param="alpha"]'),
  threshold: document.querySelector('[data-param="threshold"]'),
  threshold1: document.getElementById("threshold1"),
  threshold2: document.getElementById("threshold2"),
  bilateral: document.querySelector('[data-param="bilateral"]'),
};

function populateMethods() {
  const group = groupSel.value;
  methodSel.innerHTML = "";
  GROUP_METHODS[group].forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.value;
    opt.textContent = m.label;
    methodSel.appendChild(opt);
  });
  updateParamVisibility();
}

function updateParamVisibility() {
  const group = groupSel.value;
  const method = methodSel.value;

  paramFields.sigma.style.display = "none";
  paramFields.amount.style.display = "none";
  paramFields.alpha.style.display = "none";
  document.querySelector('[data-param="threshold"]').style.display = "none";
  paramFields.threshold1.parentElement.style.display = "none";
  paramFields.threshold2.parentElement.style.display = "none";
  paramFields.bilateral.style.display = "none";

  if (group === "blur") {
    if (method === "gaussian") paramFields.sigma.style.display = "flex";
    if (method === "median") {/* no extra */}
    if (method === "bilateral") paramFields.bilateral.style.display = "flex";
  }

  if (group === "sharpen") {
    if (method === "unsharp") {
      paramFields.sigma.style.display = "flex";
      paramFields.amount.style.display = "flex";
    } else if (method === "laplacian_sharpen") {
      paramFields.alpha.style.display = "flex";
    }
  }

  if (group === "edge") {
    if (method === "canny") {
      paramFields.threshold1.parentElement.style.display = "flex";
      paramFields.threshold2.parentElement.style.display = "flex";
    } else if (method === "binary") {
      document.querySelector('[data-param="threshold"]').style.display = "flex";
    }
  }
}

function applyPreset() {
  const v = presetsSel.value;
  if (!v) return;

  if (v === "sharpen_blurry") {
    // "Brighten & Sharpen"
    // We'll do unsharp in Basic; brightness handled by gamma in Advanced when switching
    groupSel.value = "sharpen";
    populateMethods();
    methodSel.value = "unsharp";
    document.getElementById("kernel_size").value = 5;
    document.getElementById("sigma").value = 1.2;
    document.getElementById("amount").value = 1.6;
  } else if (v === "soft_skin") {
    groupSel.value = "blur";
    populateMethods();
    methodSel.value = "median";
    document.getElementById("kernel_size").value = 5;
  } else if (v === "edge_boost") {
    groupSel.value = "edge";
    populateMethods();
    methodSel.value = "canny";
    document.getElementById("threshold1").value = 50;
    document.getElementById("threshold2").value = 150;
  } else if (v === "auto_contrast") {
    // Switch to advanced histogram equalization
    selectTab("advanced");
    advMethodSel.value = "hist_eq";
  } else if (v === "low_light") {
    selectTab("advanced");
    advMethodSel.value = "gamma";
    document.getElementById("gamma").value = 0.6;
    document.getElementById("c").value = 1.0;
  }
  updateParamVisibility();
  debouncedProcess();
}

async function doProcess() {
  const fd = new FormData(form);
  fd.set("want_json", "1");

  // Respect which tab is active: if Advanced, force group=advanced + selected method
  if (panelAdvanced.classList.contains("hidden")) {
    // Basic
    fd.set("group", groupSel.value);
    fd.set("method", methodSel.value);
  } else {
    fd.set("group", "advanced");
    fd.set("method", advMethodSel.value);
  }

  // Histograms
  fd.set("show_hist", showHist.checked ? "1" : "0");

  if (!imageInput.files.length && originalPath.value) {
    fd.delete("image");
  }

  procBtn.disabled = true;
  procBtn.textContent = "Processing...";

  try {
    const res = await fetch("/process", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.original_url) {
      originalPath.value = data.original_url;
      originalPreview.src = data.original_url + "?t=" + Date.now();
    }
    if (data.processed_url) {
      processedPreview.src = data.processed_url + "?t=" + Date.now();
      downloadLink.href = data.processed_url;
      const fileName = data.processed_url.split("/").pop();
      downloadLink.download = fileName;
    }

    if (data.hist_before && data.hist_after) {
      histArea.classList.remove("hidden");
      histBefore.src = data.hist_before;
      histAfter.src = data.hist_after;
    } else {
      histArea.classList.add("hidden");
    }

  } catch (e) {
    alert("Processing failed: " + e.message);
  } finally {
    procBtn.disabled = false;
    procBtn.textContent = "Process";
  }
}

let procTimer = null;
function debouncedProcess() {
  clearTimeout(procTimer);
  procTimer = setTimeout(() => doProcess(), 250);
}

// Tabs
function selectTab(name) {
  tabs.forEach(t => t.classList.remove("active"));
  if (name === "basic") {
    tabs[0].classList.add("active");
    panelBasic.classList.remove("hidden");
    panelAdvanced.classList.add("hidden");
  } else {
    tabs[1].classList.add("active");
    panelAdvanced.classList.remove("hidden");
    panelBasic.classList.add("hidden");
  }
  debouncedProcess();
}
tabs[0].addEventListener("click", () => selectTab("basic"));
tabs[1].addEventListener("click", () => selectTab("advanced"));

// Events
groupSel.addEventListener("change", () => { populateMethods(); debouncedProcess(); });
methodSel.addEventListener("change", () => { updateParamVisibility(); debouncedProcess(); });
advMethodSel.addEventListener("change", () => debouncedProcess());
presetsSel.addEventListener("change", applyPreset);

form.addEventListener("input", (e) => {
  if (["image"].includes(e.target.id)) return;
  debouncedProcess();
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  doProcess();
});

imageInput.addEventListener("change", () => {
  originalPath.value = "";
  doProcess();
});

// Initial
populateMethods();
updateParamVisibility();


// ---- Smart Presets ----
const presetButtons = document.querySelectorAll(".preset");
const stagesWrap = document.getElementById("stages");
const stagesGrid = document.getElementById("stagesGrid");
const recoBar = document.getElementById("preset-reco");
const recoName = document.getElementById("preset-reco-name");

async function doPreset(presetName) {
  const fd = new FormData(form);
  fd.set("want_json", "1");
  fd.set("preset", presetName);
  fd.set("save_intermediates", "1");
  fd.set("show_hist", showHist && showHist.checked ? "1" : "0");

  if (!imageInput.files.length && originalPath.value) {
    fd.delete("image");
  }

  procBtn.disabled = true;
  procBtn.textContent = "Processing...";

  try {
    const res = await fetch("/preset", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (data.original_url) {
      originalPath.value = data.original_url;
      originalPreview.src = data.original_url + "?t=" + Date.now();
    }
    if (data.processed_url) {
      processedPreview.src = data.processed_url + "?t=" + Date.now();
      downloadLink.href = data.processed_url;
      const fileName = data.processed_url.split("/").pop();
      downloadLink.download = fileName;
    }

    // Stages
    stagesGrid.innerHTML = "";
    if (data.stages && data.stages.length) {
      stagesWrap.classList.remove("hidden");
      data.stages.forEach(s => {
        const card = document.createElement("div");
        card.className = "stage-card";
        const h5 = document.createElement("h5");
        h5.textContent = s.label;
        const img = document.createElement("img");
        img.src = s.url + "?t=" + Date.now();
        card.appendChild(h5);
        card.appendChild(img);
        stagesGrid.appendChild(card);
      });
    } else {
      stagesWrap.classList.add("hidden");
    }

    if (data.hist_before && data.hist_after) {
      histArea.classList.remove("hidden");
      histBefore.src = data.hist_before;
      histAfter.src = data.hist_after;
    }

  } catch (e) {
    alert("Preset failed: " + e.message);
  } finally {
    procBtn.disabled = false;
    procBtn.textContent = "Process";
  }
}

presetButtons.forEach(btn => {
  btn.addEventListener("click", () => doPreset(btn.dataset.preset));
});

async function fetchSuggestion() {
  const fd = new FormData();
  if (imageInput.files.length) {
    fd.set("image", imageInput.files[0]);
  } else if (originalPath.value) {
    fd.set("original_path", originalPath.value);
  } else {
    return;
  }
  try {
    const res = await fetch("/suggest", { method: "POST", body: fd });
    const data = await res.json();
    if (data.recommended_preset) {
      recoBar.classList.remove("hidden");
      recoName.textContent = data.recommended_preset.replace("_", " ");
    }
  } catch {}
}

imageInput.addEventListener("change", fetchSuggestion);
