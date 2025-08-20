let trainingInterval = null;

document.getElementById("startBtn").addEventListener("click", () => {
    const metrics = Array.from(document.querySelectorAll("#controls input[type=checkbox]:checked"))
                         .map(cb => cb.value);

    fetch("/start_training", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({metrics, max_steps: 50})
    }).then(() => {
        trainingInterval = setInterval(fetchStep, 500);
    });
});

document.getElementById("stopBtn").addEventListener("click", () => {
    fetch("/stop_training", {method: "POST"});
    clearInterval(trainingInterval);
});

function fetchStep() {
    fetch("/training_step").then(r => r.json()).then(data => {
        if (data.status === "stopped") {
            clearInterval(trainingInterval);
            return;
        }
        document.getElementById("output").innerHTML +=
            `<p>Step ${data.step} | Loss: ${data.loss} | FFN: ${data.ffn_output}</p>`;
    });
}

