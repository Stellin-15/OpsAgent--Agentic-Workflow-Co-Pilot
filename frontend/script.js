document.addEventListener('DOMContentLoaded', () => {
    // Get references to all the HTML elements we'll need
    const ticketForm = document.getElementById('ticket-form');
    const titleInput = document.getElementById('ticket-title');
    const descriptionInput = document.getElementById('ticket-description');
    const submitButton = document.getElementById('submit-button');
    
    const responseSection = document.getElementById('response-section');
    const draftReplyEl = document.getElementById('draft-reply');
    const approveButton = document.getElementById('approve-button');
    const rejectButton = document.getElementById('reject-button');
    
    const statusMessageEl = document.getElementById('status-message');

    let currentTicketId = null;

    // --- Event Listener for the form submission ---
    ticketForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent the browser from reloading the page

        // Show a "thinking" message
        submitButton.disabled = true;
        submitButton.textContent = 'Generating...';
        statusMessageEl.textContent = '';
        responseSection.classList.add('hidden');

        // Create a random ticket ID for this session
        currentTicketId = Math.floor(Math.random() * 100000);

        try {
            // Send the ticket data to our backend API
            const response = await fetch('/api/tickets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    id: currentTicketId,
                    title: titleInput.value,
                    description: descriptionInput.value,
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }

            const data = await response.json();

            // Display the draft reply
            draftReplyEl.textContent = data.draft_reply;
            responseSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error submitting ticket:', error);
            showStatusMessage('Error: Could not generate draft. Check console for details.', 'error');
        } finally {
            // Re-enable the submit button
            submitButton.disabled = false;
            submitButton.textContent = 'Generate Draft Reply';
        }
    });

    // --- Event Listener for the Approve button ---
    approveButton.addEventListener('click', async () => {
        if (!currentTicketId) return;

        showStatusMessage('Approving and sending to Slack...', 'info');
        
        try {
            // Call the /api/approve endpoint
            const response = await fetch(`/api/approve/${currentTicketId}`);
            
            if (!response.ok) {
                throw new Error('Approval request failed.');
            }

            const data = await response.json();

            if (data.status === 'draft_approved_and_sent') {
                showStatusMessage('Successfully approved and sent to Slack!', 'success');
                // Hide the response section and clear the form for the next ticket
                responseSection.classList.add('hidden');
                ticketForm.reset();
            } else {
                throw new Error(data.message || 'Unknown approval error.');
            }

        } catch (error) {
            console.error('Error approving draft:', error);
            showStatusMessage(`Error: ${error.message}`, 'error');
        }
    });
    
    // --- Event Listener for the Reject button ---
    rejectButton.addEventListener('click', () => {
        showStatusMessage('Draft rejected. You can submit a new ticket.', 'info');
        responseSection.classList.add('hidden');
        ticketForm.reset();
        currentTicketId = null;
    });

    // Helper function to show status messages
    function showStatusMessage(message, type) {
        statusMessageEl.textContent = message;
        statusMessageEl.style.backgroundColor = type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : '#cce5ff';
        statusMessageEl.style.color = type === 'success' ? '#155724' : type === 'error' ? '#721c24' : '#004085';
    }
});