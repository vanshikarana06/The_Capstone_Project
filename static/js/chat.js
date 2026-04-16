async function sendChatMessage() {
    const inputField = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const loader = document.getElementById('ai-loader');
    
    const message = inputField.value.trim();
    if (!message) return;

    // 1. UI Update: Show User Message & Clear Input
    chatWindow.innerHTML += `<div class="user-msg"><b>You:</b> ${message}</div>`;
    inputField.value = '';
    
    // 2. UX: Show the "Thinking" State
    loader.classList.remove('hidden');
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom

    try {
        // 3. THE FETCH CALL: Talking to your FastAPI /chat endpoint
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // 4. Update UI with AI Response
        if (data.analysis_reply) {
            chatWindow.innerHTML += `<div class="ai-msg"><b>AI:</b> ${data.analysis_reply}</div>`;
        } else {
            chatWindow.innerHTML += `<div class="error-msg">Error: ${data.error}</div>`;
        }
    } catch (err) {
        console.error("Connection failed:", err);
    } finally {
        // 5. UX: Hide the Spinner regardless of success or failure
        loader.classList.add('hidden');
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}