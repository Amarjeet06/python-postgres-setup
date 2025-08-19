// Basic functionality for the chat interface
document.addEventListener('DOMContentLoaded', function() {
    const chatHistory = document.getElementById('chatHistory');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.querySelector('.btn-clear');

    // Auto-resize textarea as user types
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Send message when button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Send message when Enter is pressed (but allow Shift+Enter for new lines)
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Clear chat history
    clearButton.addEventListener('click', function() {
        chatHistory.innerHTML = `
            <div class="message gpt-message">
                <div class="avatar"><i class="fas fa-robot"></i></div>
                <div class="content">
                    <p>Hello! I'm your custom GPT assistant. How can I help you today?</p>
                </div>
            </div>
        `;
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';
        userInput.style.height = 'auto';

        // Simulate GPT typing indicator
        const typingIndicator = addTypingIndicator();
        
        // In a real app, you would send the message to your backend here
        // For now, we'll simulate a response after a delay
        setTimeout(() => {
            removeTypingIndicator(typingIndicator);
            // This is where you would receive the actual GPT response
            const response = "This is a simulated response. In the real app, this would come from your GPT backend connected to PostgreSQL.";
            addMessage(response, 'gpt');
        }, 1000);
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.innerHTML = `
            <div class="avatar"><i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i></div>
            <div class="content">
                <p>${text}</p>
            </div>
        `;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message gpt-message typing-indicator';
        typingDiv.innerHTML = `
            <div class="avatar"><i class="fas fa-robot"></i></div>
            <div class="content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        chatHistory.appendChild(typingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return typingDiv;
    }

    function removeTypingIndicator(element) {
        element.remove();
    }
});