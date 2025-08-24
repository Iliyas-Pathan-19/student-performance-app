document.getElementById("predictionForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = {
    StudyHours: parseFloat(document.getElementById("studyHours").value),
    SleepHours: parseFloat(document.getElementById("sleepHours").value),
    PreviousMarks: parseFloat(document.getElementById("previousMarks").value),
    PapersPrepared: parseFloat(document.getElementById("papersPrepared").value),
    Age: parseFloat(document.getElementById("age").value),
    Sex: document.getElementById("sex").value
  };

  const response = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });

  const result = await response.json();
  if (result.predicted_grade) {
    document.getElementById("result").innerText = `Predicted Grade: ${result.predicted_grade}`;
  } else {
    document.getElementById("result").innerText = `Error: ${result.error}`;
  }
});
