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
  const annotatedContainer = document.getElementById("annotated-container");
  const annotatedPreview = document.getElementById("annotated-preview");
  const magnifierLens = document.getElementById("magnifier-lens");
  const spotBadge = document.getElementById("spot-badge");
  const spotStatusText = document.getElementById("spot-status-text");
  const viewHighlightedBtn = document.getElementById("view-highlighted-btn");
  const viewPreprocessedBtn = document.getElementById("view-preprocessed-btn");
  const viewOriginalBtn = document.getElementById("view-original-btn");

  // EPPO / USDA PDI & Co-Infection Elements
  const pdiMeterCard = document.getElementById("pdi-meter-card");
  const pdiGradeBadge = document.getElementById("pdi-grade-badge");
  const pdiActionText = document.getElementById("pdi-action-text");
  const coAlertsContainer = document.getElementById("co-alerts-container");
  const coAlertsList = document.getElementById("co-alerts-list");

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
    if (magnifierLens) magnifierLens.style.display = "none";
    if (pdiMeterCard) pdiMeterCard.style.display = "none";
    if (coAlertsContainer) coAlertsContainer.style.display = "none";
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

  // 4b. 3x Optical Loupe Magnifier
  if (annotatedContainer && magnifierLens && annotatedPreview) {
    function moveLoupe(e) {
      if (!annotatedPreview.src || !annotatedPreview.complete || annotatedPreview.naturalWidth === 0) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;

      const imgRect = annotatedPreview.getBoundingClientRect();
      const contRect = annotatedContainer.getBoundingClientRect();

      const relX = clientX - imgRect.left;
      const relY = clientY - imgRect.top;

      if (relX < 0 || relX > imgRect.width || relY < 0 || relY > imgRect.height) {
        magnifierLens.style.display = "none";
        return;
      }

      magnifierLens.style.display = "block";
      const lensX = clientX - contRect.left;
      const lensY = clientY - contRect.top;
      magnifierLens.style.left = `${lensX}px`;
      magnifierLens.style.top = `${lensY}px`;

      const zoom = 3.0;
      magnifierLens.style.backgroundImage = `url("${annotatedPreview.src}")`;
      magnifierLens.style.backgroundSize = `${imgRect.width * zoom}px ${imgRect.height * zoom}px`;
      const bgX = relX * zoom - 65;
      const bgY = relY * zoom - 65;
      magnifierLens.style.backgroundPosition = `-${bgX}px -${bgY}px`;
    }

    annotatedContainer.addEventListener("mousemove", moveLoupe);
    annotatedContainer.addEventListener("touchmove", moveLoupe, { passive: true });
    annotatedContainer.addEventListener("mouseleave", () => {
      magnifierLens.style.display = "none";
    });
    annotatedContainer.addEventListener("touchend", () => {
      magnifierLens.style.display = "none";
    });
  }

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

    const exportBtn = document.getElementById("export-report-btn");
    if (exportBtn) exportBtn.style.display = "inline-flex";

    preprocessedImageSrc = data.preprocessed_image_base64 || "";
    if (viewPreprocessedBtn) {
      viewPreprocessedBtn.style.display = preprocessedImageSrc ? "inline-block" : "none";
    }

    // Render Abiotic Stress & Senescence Assessment
    if (data.abiotic) {
      renderAbioticPanel(data.abiotic);
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
      renderPdiMeter(data.pdi_rating);
      renderCoAlerts(data.co_alerts);
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

      renderPdiMeter(null);
      renderCoAlerts(null);

      // Render top plant identification candidates
      renderBotanicalCandidatesList(data.species.top_candidates || []);
    }
  }

  function renderPdiMeter(pdi) {
    if (!pdiMeterCard) return;
    if (!pdi) {
      pdiMeterCard.style.display = "none";
      return;
    }
    pdiMeterCard.style.display = "flex";
    if (pdiGradeBadge) {
      pdiGradeBadge.textContent = pdi.label;
      pdiGradeBadge.style.background = pdi.color;
    }
    if (pdiActionText) {
      pdiActionText.textContent = pdi.clinical_action;
    }
    const segments = document.querySelectorAll(".pdi-segment");
    segments.forEach(seg => {
      const g = parseInt(seg.dataset.grade, 10);
      if (g === pdi.grade) {
        seg.classList.add("active");
      } else {
        seg.classList.remove("active");
      }
    });
  }

  function renderCoAlerts(alerts) {
    if (!coAlertsContainer || !coAlertsList) return;
    if (!alerts || alerts.length === 0) {
      coAlertsContainer.style.display = "none";
      return;
    }
    coAlertsContainer.style.display = "flex";
    coAlertsList.innerHTML = alerts.map(a => `
      <div class="co-alert-item">
        <div class="co-alert-top">
          <span class="co-alert-badge">${a.badge}</span>
          <span class="co-alert-type">${a.type}</span>
        </div>
        <p class="co-alert-msg">${a.message}</p>
      </div>
    `).join("");
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
        if (p.is_healthy) {
          renderPdiMeter({
            grade: 0,
            label: "Grade 0: Optimal Foliar Health (Clean)",
            color: "#10b981",
            clinical_action: "No pathological interventions required. Maintain routine botanical care."
          });
          renderCoAlerts([]);
        }
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

  // 6. Abiotic Stress & Foliar Senescence Panel Renderer
  function renderAbioticPanel(abiotic) {
    const abioticCard = document.getElementById("abiotic-card");
    if (!abioticCard) return;
    abioticCard.style.display = "block";

    const badgeWater = document.getElementById("badge-water");
    const badgeAging = document.getElementById("badge-aging");
    const summaryText = document.getElementById("abiotic-summary-text");

    const valVigor = document.getElementById("val-vigor");
    const barVigor = document.getElementById("bar-vigor");
    const valChlorosis = document.getElementById("val-chlorosis");
    const barChlorosis = document.getElementById("bar-chlorosis");
    const valCrispness = document.getElementById("val-crispness");
    const barCrispness = document.getElementById("bar-crispness");

    const advBox = document.getElementById("abiotic-advisory-box");
    const advList = document.getElementById("abiotic-advisory-list");

    if (badgeWater) {
      badgeWater.textContent = abiotic.water_status;
      badgeWater.className = "abiotic-badge " + (
        abiotic.water_severity === "Normal" ? "optimal" :
        abiotic.water_severity === "Moderate" ? "warning" : "severe"
      );
    }

    if (badgeAging) {
      badgeAging.textContent = abiotic.aging_status;
      badgeAging.className = "abiotic-badge " + (
        abiotic.aging_status.includes("Natural") ? "senescence" : "optimal"
      );
    }

    if (summaryText) summaryText.textContent = abiotic.status_summary;

    if (valVigor && barVigor) {
      valVigor.textContent = `${abiotic.vigor_pct}%`;
      barVigor.style.width = `${abiotic.vigor_pct}%`;
    }
    if (valChlorosis && barChlorosis) {
      valChlorosis.textContent = `${abiotic.chlorosis_pct}%`;
      barChlorosis.style.width = `${Math.min(abiotic.chlorosis_pct, 100)}%`;
    }
    if (valCrispness && barCrispness) {
      valCrispness.textContent = `${abiotic.crispness_pct}%`;
      barCrispness.style.width = `${Math.min(abiotic.crispness_pct, 100)}%`;
    }

    if (advBox && advList && abiotic.abiotic_actions && abiotic.abiotic_actions.length > 0) {
      advList.innerHTML = abiotic.abiotic_actions.map(act => `<li>${act}</li>`).join("");
      advBox.style.display = "block";
    } else if (advBox) {
      advBox.style.display = "none";
    }
  }

  // 7. Export Clinical Pathology Report / Lab Certificate
  const exportBtn = document.getElementById("export-report-btn");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      window.print();
    });
  }

  // 8. Clinical Pathology Encyclopedia (31 Conditions)
  const openEncyclopediaBtn = document.getElementById("open-encyclopedia-btn");
  const closeEncyclopediaBtn = document.getElementById("close-encyclopedia-btn");
  const encyclopediaModal = document.getElementById("encyclopedia-modal");
  const encyclopediaGrid = document.getElementById("encyclopedia-grid");
  const encyclopediaSearch = document.getElementById("encyclopedia-search");
  const encChips = document.querySelectorAll(".enc-chip");

  let encyclopediaData = null;
  let activeFilter = "all";

  async function loadEncyclopedia() {
    if (encyclopediaData) return;
    try {
      const res = await fetch("/api/encyclopedia");
      const data = await res.json();
      if (data.status === "success") {
        encyclopediaData = data.entries;
        renderEncyclopediaCards();
      }
    } catch (err) {
      if (encyclopediaGrid) {
        encyclopediaGrid.innerHTML = `<p class="modal-err">Failed to load clinical encyclopedia: ${err.message}</p>`;
      }
    }
  }

  function renderEncyclopediaCards() {
    if (!encyclopediaData || !encyclopediaGrid) return;
    const query = (encyclopediaSearch ? encyclopediaSearch.value : "").toLowerCase().trim();

    const keys = Object.keys(encyclopediaData);
    const filteredKeys = keys.filter(k => {
      const item = encyclopediaData[k];
      const matchesSearch = (
        item.condition_name.toLowerCase().includes(query) ||
        item.plant_name.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query) ||
        k.toLowerCase().includes(query)
      );

      if (!matchesSearch) return false;

      if (activeFilter === "universal") {
        return k.startsWith("Universal_") && !item.is_healthy;
      } else if (activeFilter === "specialized") {
        return !k.startsWith("Universal_");
      } else if (activeFilter === "healthy") {
        return item.is_healthy;
      }
      return true;
    });

    if (filteredKeys.length === 0) {
      encyclopediaGrid.innerHTML = `<div class="empty-encyclopedia">No conditions match "${query}". Try searching for Aphids, Thrips, Rust, Mildew, Scale, Cactus, etc.</div>`;
      return;
    }

    encyclopediaGrid.innerHTML = filteredKeys.map(k => {
      const item = encyclopediaData[k];
      const severityClass = item.is_healthy ? "healthy" : (item.severity.toLowerCase().includes("high") ? "severe" : "moderate");
      return `
        <div class="enc-card">
          <div class="enc-card-header">
            <div>
              <span class="enc-card-plant">${item.plant_name}</span>
              <h4 class="enc-card-title">${item.condition_name}</h4>
              <span class="enc-card-taxa">${item.scientific_name || k}</span>
            </div>
            <span class="enc-sev-pill ${severityClass}">${item.severity}</span>
          </div>
          <p class="enc-card-summary">${item.summary || "Clinical diagnostic protocol active."}</p>
          
          <div class="enc-section-block">
            <strong>👁️ Primary Visual Symptoms:</strong>
            <ul>${(item.symptoms || []).slice(0, 3).map(s => `<li>${s}</li>`).join("")}</ul>
          </div>

          <div class="enc-treatment-row">
            <div class="enc-treat-box organic">
              <strong>🌿 Organic Protocol:</strong>
              <ul>${(item.treatment?.organic || []).slice(0, 2).map(t => `<li>${t}</li>`).join("")}</ul>
            </div>
            <div class="enc-treat-box chemical">
              <strong>🧪 Chemical Protocol:</strong>
              <ul>${(item.treatment?.chemical || []).slice(0, 2).map(t => `<li>${t}</li>`).join("")}</ul>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  if (openEncyclopediaBtn) {
    openEncyclopediaBtn.addEventListener("click", () => {
      if (encyclopediaModal) encyclopediaModal.style.display = "flex";
      loadEncyclopedia();
    });
  }

  if (closeEncyclopediaBtn) {
    closeEncyclopediaBtn.addEventListener("click", () => {
      if (encyclopediaModal) encyclopediaModal.style.display = "none";
    });
  }

  if (encyclopediaModal) {
    encyclopediaModal.addEventListener("click", e => {
      if (e.target === encyclopediaModal) encyclopediaModal.style.display = "none";
    });
  }

  if (encyclopediaSearch) {
    encyclopediaSearch.addEventListener("input", renderEncyclopediaCards);
  }

  encChips.forEach(chip => {
    chip.addEventListener("click", () => {
      encChips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      activeFilter = chip.dataset.filter;
      renderEncyclopediaCards();
    });
  });
});

