document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const clearBtn = document.getElementById("clear-btn");
  const diagnoseBtn = document.getElementById("diagnose-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");
  const plantSelect = document.getElementById("plant-select");

  const resultsEmpty = document.getElementById("results-empty");
  const resultsContent = document.getElementById("results-content");
  const latencyTag = document.getElementById("latency-tag");

  // Spot visualizer
  const annotatedPreview = document.getElementById("annotated-preview");
  const spotBadge = document.getElementById("spot-badge");
  const spotStatusText = document.getElementById("spot-status-text");
  const viewHighlightedBtn = document.getElementById("view-highlighted-btn");
  const viewPreprocessedBtn = document.getElementById("view-preprocessed-btn");
  const viewOriginalBtn = document.getElementById("view-original-btn");

  // Verdict Banner
  const verdictBanner = document.getElementById("verdict-banner");
  const statusBadge = document.getElementById("status-badge");
  const verdictConfidence = document.getElementById("verdict-confidence");
  const verdictPlant = document.getElementById("verdict-plant");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictSummary = document.getElementById("verdict-summary");

  // Care Lists
  const listImmediate = document.getElementById("list-immediate");
  const listTreatmentOrganic = document.getElementById("list-treatment-organic");
  const listTreatmentChemical = document.getElementById("list-treatment-chemical");
  const listPrevention = document.getElementById("list-prevention");
  const listSymptoms = document.getElementById("list-symptoms");
  const listCauses = document.getElementById("list-causes");

  const predictionList = document.getElementById("prediction-list");

  // Layer 1 Botanical Banner & Unavailable State Elements
  const layer1IdBanner = document.getElementById("layer1-id-banner");
  const layer1SpeciesName = document.getElementById("layer1-species-name");
  const layer1Confidence = document.getElementById("layer1-confidence");
  const layer1StatusPill = document.getElementById("layer1-status-pill");

  const unavailableBanner = document.getElementById("unavailable-banner");
  const unavailablePlantText = document.getElementById("unavailable-plant-text");
  const unavailableDescription = document.getElementById("unavailable-description");
  const careSection = document.getElementById("care-section");
  const breakdownTitle = document.getElementById("breakdown-title");
  const supportedChips = document.querySelectorAll(".supp-chip");

  const systemStatus = document.getElementById("system-status");
  const statusText = document.getElementById("status-text");
  const deviceVal = document.getElementById("device-val");

  let selectedFile = null;
  let originalImageSrc = "";
  let annotatedImageSrc = "";
  let preprocessedImageSrc = "";

  // 1. Fetch system info
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error("API offline");
      const data = await res.json();

      if (data.model_ready) {
        statusText.textContent = "Model Active (Hierarchical)";
        systemStatus.style.background = "#ecfdf5";
        systemStatus.style.color = "#047857";
      } else {
        statusText.textContent = "Training Pending";
        systemStatus.style.background = "#fffbeb";
        systemStatus.style.color = "#b45309";
      }

      if (data.device) {
        deviceVal.textContent = data.device.toUpperCase().includes("CUDA")
          ? "NVIDIA RTX 3060 (CUDA)"
          : "CPU";
      }
    } catch (err) {
      statusText.textContent = "Server Offline";
      systemStatus.style.background = "#fef2f2";
      systemStatus.style.color = "#b91c1c";
    }
  }

  fetchSystemInfo();

  // 2. Drag & Drop Handlers
  ["dragenter", "dragover"].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });

  dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", e => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      originalImageSrc = e.target.result;
      imagePreview.src = originalImageSrc;
      dropPrompt.style.display = "none";
      previewContainer.style.display = "block";
      diagnoseBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener("click", e => {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    originalImageSrc = "";
    annotatedImageSrc = "";
    preprocessedImageSrc = "";
    fileInput.value = "";
    imagePreview.src = "";
    dropPrompt.style.display = "block";
    previewContainer.style.display = "none";
    diagnoseBtn.disabled = true;
    resultsEmpty.style.display = "flex";
    resultsContent.style.display = "none";
    latencyTag.style.display = "none";
  }

  // 3. Tab Navigation
  const tabButtons = document.querySelectorAll(".care-tab-btn");
  const tabPanes = document.querySelectorAll(".care-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) targetPane.classList.add("active");
    });
  });

  // 4. Spot View Toggles
  viewHighlightedBtn.addEventListener("click", () => {
    viewHighlightedBtn.classList.add("active");
    viewOriginalBtn.classList.remove("active");
    if (viewPreprocessedBtn) viewPreprocessedBtn.classList.remove("active");
    annotatedPreview.src = annotatedImageSrc || originalImageSrc;
  });

  if (viewPreprocessedBtn) {
    viewPreprocessedBtn.addEventListener("click", () => {
      viewPreprocessedBtn.classList.add("active");
      viewHighlightedBtn.classList.remove("active");
      viewOriginalBtn.classList.remove("active");
      annotatedPreview.src = preprocessedImageSrc || originalImageSrc;
    });
  }

  viewOriginalBtn.addEventListener("click", () => {
    viewOriginalBtn.classList.add("active");
    viewHighlightedBtn.classList.remove("active");
    if (viewPreprocessedBtn) viewPreprocessedBtn.classList.remove("active");
    annotatedPreview.src = originalImageSrc;
  });

  // 5. Diagnosis Execution
  diagnoseBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    diagnoseBtn.disabled = true;
    btnSpinner.style.display = "inline-block";
    btnText.textContent = "Scanning Spots & Pathology...";

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("plant", plantSelect.value);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Diagnosis failed");
      }

      displayResults(data);
    } catch (err) {
      alert(`Diagnostic Error: ${err.message}`);
    } finally {
      diagnoseBtn.disabled = false;
      btnSpinner.style.display = "none";
      btnText.textContent = "Diagnose & Detect Disease Spots";
    }
  });

  // Supported Species Chips click handler
  supportedChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const plantVal = chip.dataset.plant;
      if (plantVal && plantSelect) {
        plantSelect.value = plantVal;
        if (selectedFile) {
          diagnoseBtn.click();
        }
      }
    });
  });

  function displayResults(data) {
    resultsEmpty.style.display = "none";
    resultsContent.style.display = "flex";

    latencyTag.textContent = `${data.inference_time_ms} ms`;
    latencyTag.style.display = "inline-block";

    preprocessedImageSrc = data.preprocessed_image_base64 || "";
    if (viewPreprocessedBtn) {
      viewPreprocessedBtn.style.display = preprocessedImageSrc ? "inline-block" : "none";
    }

    // 1. Render Layer 1 Botanical Identification Strip
    if (layer1SpeciesName && data.species) {
      layer1SpeciesName.textContent = data.species.name;
      layer1Confidence.textContent = `${data.species.confidence}% Match`;
    }

    // 2. Branch: Supported vs Unsupported
    if (data.disease_available) {
      if (layer1StatusPill) {
        layer1StatusPill.className = "layer1-status-pill supported";
        layer1StatusPill.textContent = "Supported in Disease Model";
      }

      if (unavailableBanner) unavailableBanner.style.display = "none";
      if (verdictBanner) verdictBanner.style.display = "block";
      if (careSection) careSection.style.display = "block";
      if (breakdownTitle) {
        breakdownTitle.textContent = "Intra-Group Pathology Breakdown (Click to inspect care guide)";
      }

      // Spot visualization
      annotatedImageSrc = data.spots.annotated_image_base64;
      annotatedPreview.src = annotatedImageSrc;
      viewHighlightedBtn.classList.add("active");
      viewOriginalBtn.classList.remove("active");
      spotStatusText.textContent = data.spots.status_message;

      if (data.diagnosis.is_healthy || !data.spots.has_spots) {
        spotBadge.className = "spot-badge clean";
        spotBadge.textContent = "0 SPOTS (CLEAN & HEALTHY)";
      } else {
        spotBadge.className = "spot-badge spots-found";
        spotBadge.textContent = `${data.spots.spots_count} SPOTS DETECTED (${data.spots.infected_area_pct}% AREA)`;
      }

      renderDiagnosisView(data.species, data.diagnosis, data.care_guide);
      renderPredictionsList(data.predictions, data.species);
    } else {
      // Plant has NO training data in disease model
      if (layer1StatusPill) {
        layer1StatusPill.className = "layer1-status-pill unsupported";
        layer1StatusPill.textContent = "Disease Model Unavailable";
      }

      if (unavailableBanner) unavailableBanner.style.display = "block";
      if (verdictBanner) verdictBanner.style.display = "none";
      if (careSection) careSection.style.display = "none";
      if (breakdownTitle) {
        breakdownTitle.textContent = "Layer 1 Botanical Plant Identification Candidates";
      }

      if (unavailablePlantText) {
        unavailablePlantText.textContent = `Identified: ${data.species.name} (${data.species.confidence}% match). Disease training data is unavailable for this species.`;
      }
      if (unavailableDescription) {
        unavailableDescription.textContent = data.status_message;
      }

      // Spot visualizer shows original photo with inactive badge
      annotatedImageSrc = data.spots.annotated_image_base64 || originalImageSrc;
      annotatedPreview.src = annotatedImageSrc;
      viewHighlightedBtn.classList.remove("active");
      viewOriginalBtn.classList.add("active");
      spotBadge.className = "spot-badge inactive";
      spotBadge.textContent = "LESION DETECTION INACTIVE";
      spotStatusText.textContent = "Spot localization is disabled because no disease model exists for this botanical species.";

      // Render top plant identification candidates
      renderBotanicalCandidatesList(data.species.top_candidates || []);
    }
  }

  function renderDiagnosisView(species, diag, guide) {
    verdictBanner.className = "verdict-banner";
    statusBadge.className = "status-badge";

    if (diag.is_healthy) {
      verdictBanner.classList.add("healthy");
      statusBadge.classList.add("healthy");
      statusBadge.textContent = "CLEAN & HEALTHY BASELINE";
    } else if (diag.severity === "Severe") {
      verdictBanner.classList.add("diseased");
      statusBadge.classList.add("diseased");
      statusBadge.textContent = `CRITICAL: ${diag.category.toUpperCase()}`;
    } else {
      verdictBanner.classList.add("warning");
      statusBadge.classList.add("warning");
      statusBadge.textContent = `ATTENTION: ${diag.category.toUpperCase()}`;
    }

    verdictConfidence.textContent = `${diag.confidence}% Match`;
    verdictPlant.textContent = `${species.name} [${species.detected_mode}]`;
    verdictTitle.textContent = diag.condition_name;
    verdictSummary.textContent = guide.summary;

    // Tab 1: Immediate Actions
    listImmediate.innerHTML = (guide.immediate_actions || [])
      .map(act => `<li>${act}</li>`)
      .join("");

    // Tab 2: Treatment
    listTreatmentOrganic.innerHTML = (guide.treatment?.organic || [])
      .map(t => `<li>${t}</li>`)
      .join("");
    listTreatmentChemical.innerHTML = (guide.treatment?.chemical || [])
      .map(t => `<li>${t}</li>`)
      .join("");

    // Tab 3: Prevention
    listPrevention.innerHTML = (guide.prevention || [])
      .map(p => `<li>${p}</li>`)
      .join("");

    // Tab 4: Symptoms & Causes
    listSymptoms.innerHTML = (guide.symptoms || [])
      .map(s => `<li>${s}</li>`)
      .join("");
    listCauses.innerHTML = (guide.causes || [])
      .map(c => `<li>${c}</li>`)
      .join("");
  }

  function renderPredictionsList(predictions, species) {
    predictionList.innerHTML = "";

    predictions.forEach(p => {
      const item = document.createElement("div");
      item.className = "pred-item";
      item.title = "Click to inspect care and treatment guide for this condition";

      const categoryTag = p.is_healthy ? "Healthy Baseline" : p.condition_name;
      item.innerHTML = `
        <div class="pred-header">
          <span class="pred-rank-name">
            <span class="pred-rank">#${p.rank}</span> <strong>${species.name}</strong> - ${categoryTag}
          </span>
          <span class="pred-confidence">${p.confidence}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: 0%"></div>
        </div>
      `;

      item.addEventListener("click", () => {
        renderDiagnosisView(species, p, p.care_guide);
      });

      predictionList.appendChild(item);

      setTimeout(() => {
        item.querySelector(".progress-bar-fill").style.width = `${Math.min(p.confidence, 100)}%`;
      }, 60);
    });
  }

  function renderBotanicalCandidatesList(candidates) {
    predictionList.innerHTML = "";

    candidates.forEach(c => {
      const item = document.createElement("div");
      item.className = "pred-item";
      item.title = `Botanical Identification: ${c.species} (${c.confidence}%)`;

      item.innerHTML = `
        <div class="pred-header">
          <span class="pred-rank-name">
            <span class="pred-rank">#${c.rank}</span> <strong>${c.species}</strong>
          </span>
          <span class="pred-confidence">${c.confidence}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: 0%"></div>
        </div>
      `;

      predictionList.appendChild(item);

      setTimeout(() => {
        item.querySelector(".progress-bar-fill").style.width = `${Math.min(c.confidence, 100)}%`;
      }, 60);
    });
  }
});
