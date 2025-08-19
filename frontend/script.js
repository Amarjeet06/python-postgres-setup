async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (!userInput.trim()) return;

    addMessage(userInput, 'user');  // show user message
    const typingIndicator = addTypingIndicator(); // optional

    try {
        const response = await fetch('http://127.0.0.1:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: userInput })
        });

        const data = await response.json();
        removeTypingIndicator(typingIndicator);
        addMessage(data.response, 'gpt'); // show GPT reply
    } catch (error) {
        removeTypingIndicator(typingIndicator);
        addMessage("API Error: " + error.message, 'gpt');
    }
}
