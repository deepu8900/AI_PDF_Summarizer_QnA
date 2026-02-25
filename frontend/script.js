const API = "http://127.0.0.1:8000";

async function uploadPDF() {
    const file = document.getElementById("pdfFile").files[0];
    const formData = new FormData();
    formData.append("file", file);

    await fetch(`${API}/upload`, {
        method: "POST",
        body: formData
    });

    alert("Uploaded!");
}

async function getSummary() {
    const res = await fetch(`${API}/summarize`);
    const data = await res.json();
    document.getElementById("output").innerText = data.summary;
}

async function askQuestion() {
    const question = document.getElementById("question").value;

    const res = await fetch(`${API}/ask?question=${question}`, {
        method: "POST"
    });

    const data = await res.json();
    document.getElementById("output").innerText = data.answer;
}