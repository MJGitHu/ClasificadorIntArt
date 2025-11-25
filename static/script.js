document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = document.getElementById("user_text").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await response.json();

    console.log("DATA:", data); // <--- MUY útil para debug

    document.getElementById("result").innerText =
        "Resultado: " + data.prediction;

    document.getElementById("prob-student").innerText =
        "Probabilidad Estudiante: " + data.probabilities.student.toFixed(2) + "%";

    document.getElementById("prob-ai").innerText =
        "Probabilidad IA: " + data.probabilities.ai.toFixed(2) + "%";
});
