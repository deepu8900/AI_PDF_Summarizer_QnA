const API_URL = "http://127.0.0.1:8000"


async function uploadPDF(){

    const fileInput = document.getElementById("pdfFile")
    const file = fileInput.files[0]

    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(API_URL + "/upload",{
        method:"POST",
        body:formData
    })

    const data = await response.json()

    document.getElementById("uploadStatus").innerText = data.message
}


async function getSummary(){

    const response = await fetch(API_URL + "/summarize",{
        method:"POST"
    })

    const data = await response.json()

    document.getElementById("summaryResult").innerText = data.summary
}


async function askQuestion(){

    const question = document.getElementById("questionInput").value

    const response = await fetch(API_URL + "/ask?question=" + question,{
        method:"POST"
    })

    const data = await response.json()

    document.getElementById("answerResult").innerText = data.answer
}