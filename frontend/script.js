function addMessage(text, sender) {
  const chatArea = document.getElementById("chatArea");

  // create a message bubble
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message");
  if (sender === "user") {
    messageDiv.classList.add("user-message");
    messageDiv.innerText = `You: ${text}`;
  } else {
    messageDiv.classList.add("bot-message");
    messageDiv.innerText = `Bot: ${text}`;
  }

  chatArea.appendChild(messageDiv);
  chatArea.scrollTop = chatArea.scrollHeight; // auto scroll
}

async function sendMessage() {
  const input = document.getElementById("userInput");
  const userInput = input.value.trim();
  if (!userInput) return;

  addMessage(userInput, "user");
  input.value = "";

  try {
    const response = await fetch("http://127.0.0.1:8000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: userInput })
    });

    const data = await response.json();
    addMessage(data.response, "bot");
  } catch (error) {
    addMessage("API Error: " + error.message, "bot");
  }
}
