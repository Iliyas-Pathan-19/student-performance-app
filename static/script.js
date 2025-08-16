// Handle Train Form
document.getElementById("trainForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    let formData = new FormData(e.target);

    let res = await fetch("/train", {
        method: "POST",
        body: formData
    });

    let data = await res.json();
    document.getElementById("trainResult").innerHTML =
        data.error ? `<p style="color:red">${data.error}</p>` :
        `<p>✅ Model trained! Accuracy: ${data.accuracy}% <br> Features: ${data.features.join(", ")}</p>`;
});

// Handle Predict Form
document.getElementById("predictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    let inputs = {};
    [...e.target.elements].forEach(el => {
        if (el.name) inputs[el.name] = parseFloat(el.value);
    });

    let res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: inputs })
    });

    let data = await res.json();
    if (data.error) {
        document.getElementById("predictResult").innerHTML = `<p style="color:red">${data.error}</p>`;
    } else {
        document.getElementById("predictResult").innerHTML =
            `<p>📈 Predicted Marks: <b>${data.prediction}</b></p>
             <p>💡 Advice: ${data.advice.join(", ")}</p>`;
    }
});
