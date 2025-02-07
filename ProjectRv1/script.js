//The website's user input  reading and modification Javascript program.

document.getElementById("fishForm").addEventListener("submit", function(event) {
    event.preventDefault();

    //Reading the user inputs and storing them.
    let inputData = {
        "Weight": parseFloat(document.getElementById("weight").value),
        "Length1": parseFloat(document.getElementById("length1").value),
        "Length2": parseFloat(document.getElementById("length2").value),
        "Length3": parseFloat(document.getElementById("length3").value),
        "Height": parseFloat(document.getElementById("height").value),
        "Width": parseFloat(document.getElementById("width").value)
    };

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData)
    })
    .then(response => response.json())
    .then(data => {
        console.log("Received data:", data);

        //Writing the outputs from the model onto the website.
        if(data.error){
            document.getElementById("prediction").innerText = "Error: " + data.error;
        } 
        else{
            document.getElementById("prediction").innerText = "Predicted Species: " + data["Predicted Species"];
            document.getElementById("f1Score").innerText = "F1 Score: " + data.Metrics["F1 Score"].toFixed(4);
            document.getElementById("accuracy").innerText = "Accuracy: " + (data.Metrics["Accuracy"]*100).toFixed(4)  + "%";
            document.getElementById("precision").innerText = "Precision: " + (data.Metrics["Precision"]*100).toFixed(4) + "%";
        }
    })
    //Handling errors.
    .catch(error => {
        console.error("Fetch error:", error);
        document.getElementById("prediction").innerText = "Error: Could not fetch result.";
    });
});
