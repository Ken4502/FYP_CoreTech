document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const clearBtn = document.getElementById("clear-btn");
  const identifyBtn = document.getElementById("identify-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");

  const resultsEmpty = document.getElementById("results-empty");
  const resultsContent = document.getElementById("results-content");
  const topSpeciesName = document.getElementById("top-species-name");
  const topConfidenceBadge = document.getElementById("top-confidence-badge");
  const predictionList = document.getElementById("prediction-list");
  const latencyTag = document.getElementById("latency-tag");

  const systemStatus = document.getElementById("system-status");
  const statusText = document.getElementById("status-text");
  const speciesCountVal = document.getElementById("species-count-val");
  const deviceVal = document.getElementById("device-val");

  const viewToggleBar = document.getElementById("view-toggle-bar");
  const btnShowOriginal = document.getElementById("btn-show-original");
  const btnShowPreprocessed = document.getElementById("btn-show-preprocessed");
  const prepSummaryCard = document.getElementById("prep-summary-card");
  const prepThumbOrig = document.getElementById("prep-thumb-orig");
  const prepThumbClean = document.getElementById("prep-thumb-clean");

  let selectedFile = null;
  let originalImgData = "";
  let preprocessedImgData = "";

  // 1. Fetch system & model info
  async function fetchSystemInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error("API offline");
      const data = await res.json();

      if (data.model_ready) {
        statusText.textContent = "Model Ready (Active)";
        systemStatus.style.background = "#e8f5ee";
        systemStatus.style.color = "#1e5e44";
      } else {
        statusText.textContent = "Model Checkpoint Pending";
        systemStatus.style.background = "#fef3c7";
        systemStatus.style.color = "#92400e";
      }

      if (data.classes_count) {
        speciesCountVal.textContent = `${data.classes_count} Classes`;
      }
      if (data.device) {
        deviceVal.textContent = data.device.toUpperCase().includes("CUDA")
          ? "NVIDIA GPU (CUDA)"
          : "CPU";
      }
    } catch (err) {
      statusText.textContent = "Server Offline";
      systemStatus.style.background = "#fee2e2";
      systemStatus.style.color = "#991b1b";
    }
  }

  fetchSystemInfo();

  // 2. Drag and drop handlers
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
      alert("Please upload an image file (JPG, PNG, WEBP).");
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
      originalImgData = e.target.result;
      preprocessedImgData = "";
      imagePreview.src = originalImgData;
      dropPrompt.style.display = "none";
      previewContainer.style.display = "block";
      identifyBtn.disabled = false;
      if (viewToggleBar) viewToggleBar.style.display = "none";
      if (prepSummaryCard) prepSummaryCard.style.display = "none";
      if (btnShowOriginal) btnShowOriginal.classList.add("active");
      if (btnShowPreprocessed) btnShowPreprocessed.classList.remove("active");
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener("click", e => {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    originalImgData = "";
    preprocessedImgData = "";
    fileInput.value = "";
    imagePreview.src = "";
    dropPrompt.style.display = "block";
    previewContainer.style.display = "none";
    identifyBtn.disabled = true;
    resultsEmpty.style.display = "flex";
    resultsContent.style.display = "none";
    latencyTag.style.display = "none";
    if (viewToggleBar) viewToggleBar.style.display = "none";
    if (prepSummaryCard) prepSummaryCard.style.display = "none";
    if (btnShowOriginal) btnShowOriginal.classList.add("active");
    if (btnShowPreprocessed) btnShowPreprocessed.classList.remove("active");
  }

  // View toggle listeners
  if (btnShowOriginal && btnShowPreprocessed) {
    btnShowOriginal.addEventListener("click", () => {
      if (originalImgData) {
        imagePreview.src = originalImgData;
        btnShowOriginal.classList.add("active");
        btnShowPreprocessed.classList.remove("active");
      }
    });

    btnShowPreprocessed.addEventListener("click", () => {
      if (preprocessedImgData) {
        imagePreview.src = preprocessedImgData;
        btnShowPreprocessed.classList.add("active");
        btnShowOriginal.classList.remove("active");
      }
    });
  }

  // 3. Identification prediction handler
  identifyBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    identifyBtn.disabled = true;
    btnSpinner.style.display = "inline-block";
    btnText.textContent = "Analyzing Leaf...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Identification failed");
      }

      displayResults(data);
    } catch (err) {
      alert(`Identification Error: ${err.message}`);
    } finally {
      identifyBtn.disabled = false;
      btnSpinner.style.display = "none";
      btnText.textContent = "Identify Plant Species";
    }
  });

  function displayResults(data) {
    resultsEmpty.style.display = "none";
    resultsContent.style.display = "flex";

    // Preprocessed image support
    if (data.preprocessed_image_base64) {
      preprocessedImgData = data.preprocessed_image_base64.startsWith("data:")
        ? data.preprocessed_image_base64
        : `data:image/jpeg;base64,${data.preprocessed_image_base64}`;

      if (viewToggleBar) viewToggleBar.style.display = "flex";
      if (btnShowPreprocessed && btnShowOriginal) {
        btnShowPreprocessed.classList.add("active");
        btnShowOriginal.classList.remove("active");
      }
      imagePreview.src = preprocessedImgData;

      if (prepSummaryCard && prepThumbOrig && prepThumbClean) {
        prepThumbOrig.src = originalImgData;
        prepThumbClean.src = preprocessedImgData;
        prepSummaryCard.style.display = "block";
      }
    }

    // Latency
    latencyTag.textContent = `${data.inference_time_ms} ms`;
    latencyTag.style.display = "inline-block";

    const topMatch = data.predictions[0];
    topSpeciesName.textContent = topMatch.species;
    topConfidenceBadge.textContent = `${topMatch.confidence}% Confidence`;

    // Render list
    predictionList.innerHTML = "";
    data.predictions.forEach(p => {
      const item = document.createElement("div");
      item.className = "pred-item";
      item.innerHTML = `
        <div class="pred-header">
          <span class="pred-rank-name">
            <span class="pred-rank">#${p.rank}</span> ${p.species}
          </span>
          <span class="pred-confidence">${p.confidence}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: 0%"></div>
        </div>
      `;
      predictionList.appendChild(item);

      // Trigger animation
      setTimeout(() => {
        item.querySelector(".progress-bar-fill").style.width = `${Math.min(p.confidence, 100)}%`;
      }, 50);
    });
  }
});

