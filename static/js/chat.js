let isAITyping = false; // Add this global variable at the top

async function sendChatMessage() {
    const inputField = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const loader = document.getElementById('ai-loader');
    
    const message = inputField.value.trim();

    // NEW GUARD: Stop if message is empty OR if AI is already processing
    if (!message || isAITyping) return;

    isAITyping = true; // LOCK: No more clicks allowed for now
    
    chatWindow.innerHTML += `<div class="user-msg"><b>You:</b> ${message}</div>`;
    inputField.value = '';
    loader.classList.remove('hidden');
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        if (data.analysis_reply) {
            chatWindow.innerHTML += `<div class="ai-msg"><b>AI:</b> ${data.analysis_reply}</div>`;
        } else {
            chatWindow.innerHTML += `<div class="error-msg">Error: ${data.error || "Rate limit reached"}</div>`;
        }
    } catch (err) {
        console.error("Connection failed:", err);
    } finally {
        isAITyping = false; // UNLOCK: Ready for the next message
        loader.classList.add('hidden');
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}