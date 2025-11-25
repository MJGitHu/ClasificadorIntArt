document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = document.getElementById("user_text").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await response.json();
    document.getElementById("result").innerText =
        "Predicción: " + data.prediction;
});
