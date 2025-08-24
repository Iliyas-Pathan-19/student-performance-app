document.getElementById("predictForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  let formData = {};
  new FormData(this).forEach((value, key) => formData[key] = value);

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData)
  });

  const data = await res.json();
  document.getElementById("result").innerHTML =
    data.predicted_grade ? `🎯 Predicted Grade: <b>${data.predicted_grade}</b>` : `⚠️ Error: ${data.error}`;
});
