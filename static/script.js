document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = document.getElementById("user_text").value;

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const data = await response.json();

    document.getElementById("result").innerHTML = `
        <h3>Resultado: ${data.label}</h3>
        <p><strong>Probabilidad Estudiante:</strong> ${data.probabilities.student}%</p>
        <p><strong>Probabilidad IA:</strong> ${data.probabilities.ai}%</p>
    `;
});
