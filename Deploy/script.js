// 🔄 Replaces mockPredictions: Fetch live predictions from FastAPI
async function fetchPredictionsFromAPI(word, top_k = 5) {
  if (!word || word.length < 2) return [];

  try {
    const response = await fetch(
      `http://127.0.0.1:8000/predictions/?text=${encodeURIComponent(word)}&top_k=${top_k}`
    );
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Prediction API error:", error);
    return {predictions: []};
  }
}


document.addEventListener("DOMContentLoaded", function () {
        const typingInput = document.getElementById("typingInput");
        const predictionOverlay = document.getElementById("predictionOverlay");
        const predictionsRow = document.getElementById("predictionsRow");

        let currentInlinePrediction = "";

        // Function to update the inline prediction overlay
        function updateInlinePrediction() {
          const inputText = typingInput.value;
          const words = inputText.split(" ");
          const lastWord = words[words.length - 1];

          // Clear previous prediction
          currentInlinePrediction = "";
          
          if (inputText && inputText.length >= 2) {
            fetchPredictionsFromAPI(inputText, 1).then(data => {
              if (data && data.predictions && data.predictions.length > 0) {
                const prediction = data.predictions[0];
                const words = inputText.split(" ");
                const lastWord = words[words.length - 1];
                if (prediction && prediction.word.toLowerCase().startsWith(lastWord.toLowerCase())) {
                  // Show the remaining part of the prediction
                  const remainingPrediction = prediction.word.slice(lastWord.length);
                  currentInlinePrediction = remainingPrediction;
                  
                  // Create the overlay text
                  predictionOverlay.textContent = inputText + remainingPrediction;
                } else if (prediction) {
                  currentInlinePrediction = " " + prediction.word;
                  predictionOverlay.textContent = inputText + " " + prediction.word;
                }
                 else {
                  predictionOverlay.textContent = "";
                }
              } else {
                predictionOverlay.textContent = "";
              }
            });
          } else {
            predictionOverlay.textContent = "";
          }
        }

        // Function to update the predictions row
        function updatePredictionsRow(data) {
          predictionsRow.innerHTML = "";

          if (!data || !data.predictions || data.predictions.length === 0) return;

          data.predictions.forEach((prediction, index) => {
            const card = document.createElement("div");
            card.className =
              "prediction-card bg-gray-200 border border-gray-200 rounded p-3 min-w-[120px] flex flex-col items-center fade-in cursor-pointer";
            card.innerHTML = `
              <span class="font-medium text-gray-800 mb-1">${prediction.word}</span>
              <span class="text-xs text-gray-500">${(prediction.probability * 100).toFixed(1)}%</span>
            `;

            card.addEventListener("click", () => {
              insertPrediction(prediction.word);
            });

            predictionsRow.appendChild(card);
          });
        }

        // Function to insert a prediction into the input
        function insertPrediction(prediction) {
          const inputText = typingInput.value;
          const words = inputText.split(" ");
          words.pop(); // Remove the last (incomplete) word

          const newText = words.join(" ") + (words.length > 0 ? " " : "") + prediction + " ";
          typingInput.value = newText;

          // Clear predictions and update
          currentInlinePrediction = "";
          predictionOverlay.textContent = "";
          updateDisplay();
          typingInput.focus();
        }

        // Function to accept inline prediction
        function acceptInlinePrediction() {
          if (currentInlinePrediction) {
            const inputText = typingInput.value;
            
            if (currentInlinePrediction.startsWith(" ")) {
              // It's a word suggestion, not a completion
              typingInput.value = inputText + currentInlinePrediction;
            } else {
              // It's a word completion
              typingInput.value = inputText + currentInlinePrediction + " ";
            }
            
            currentInlinePrediction = "";
            predictionOverlay.textContent = "";
            updateDisplay();
          }
        }

        // Main update function
        function updateDisplay() {
          const inputText = typingInput.value;
          const words = inputText.split(" ");
          const lastWord = words[words.length - 1];

          // Update inline prediction
          updateInlinePrediction();

          // Update prediction cards
          if (inputText && inputText.length >= 2) {
            fetchPredictionsFromAPI(inputText).then(predictions => {
              updatePredictionsRow(predictions);
            });
          } else {
            updatePredictionsRow([]);
          }
        }


        // Event listeners
        typingInput.addEventListener("input", function () {
          updateDisplay();
        });

        typingInput.addEventListener("keydown", function (e) {
          if (e.key === "Tab" && currentInlinePrediction) {
            e.preventDefault();
            acceptInlinePrediction();
          }
        });

        // Initialize
        typingInput.focus();
      });
