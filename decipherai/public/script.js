window.processImage = async function () {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);

  // Show processing steps
  document.getElementById("processingSection").style.display = "block";
  document.getElementById("step1").classList.add("active");

  try {
    const res = await fetch("http://localhost:5000/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    // Set results
    document.getElementById("detectedText").textContent = data.detected_text;
    document.getElementById("translation").textContent = data.translation;
    document.getElementById("historicalContext").textContent = data.historical_context;

    // Animate steps
    setTimeout(() => {
      document.getElementById("step1").classList.remove("active");
      document.getElementById("step2").classList.add("active");
    }, 1000);
    setTimeout(() => {
      document.getElementById("step2").classList.remove("active");
      document.getElementById("step3").classList.add("active");
    }, 2000);
    setTimeout(() => {
      document.getElementById("step3").classList.remove("active");
      document.getElementById("step4").classList.add("active");
    }, 3000);
    setTimeout(() => {
      document.getElementById("step4").classList.remove("active");
      document.getElementById("step5").classList.add("active");
    }, 4000);
    setTimeout(() => {
      document.getElementById("resultsSection").style.display = "block";
    }, 5000);
  } catch (err) {
    alert("Upload failed");
    console.error(err);
  }
};

window.resetUpload = function () {
  document.getElementById("fileInput").value = "";
  document.getElementById("uploadPreview").style.display = "none";
  document.getElementById("processingSection").style.display = "none";
  document.getElementById("resultsSection").style.display = "none";

  for (let i = 1; i <= 5; i++) {
    document.getElementById(`step${i}`).classList.remove("active");
  }
};
