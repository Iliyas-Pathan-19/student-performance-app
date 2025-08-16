document.getElementById("uploadForm").onsubmit = async function(e) {
  e.preventDefault();
  let formData = new FormData(this);

  let res = await fetch("/train", {
    method: "POST",
    body: formData
  });
  let data = await res.json();

  document.getElementById("result").innerText = 
    "✅ Model trained! Accuracy: " + data.accuracy + "%";
};

async function predictGrade() {
  let age = parseFloat(document.getElementById("age").value) || 0;
  let study = parseFloat(document.getElementById("study").value) || 0;
  let sleep = parseFloat(document.getElementById("sleep").value) || 0;

  let res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ features: [age, study, sleep] })
  });

  let data = await res.json();
  document.getElementById("prediction").innerText = 
    "🎯 Predicted Grade: " + data.prediction;
}
