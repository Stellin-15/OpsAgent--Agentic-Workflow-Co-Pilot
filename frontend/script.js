document.addEventListener('DOMContentLoaded', () => {
    // Get references to all the HTML elements
    const ticketForm = document.getElementById('ticket-form');
    const titleInput = document.getElementById('ticket-title');
    const descriptionInput = document.getElementById('ticket-description');
    const submitButton = document.getElementById('submit-button');
    
    const responseSection = document.getElementById('response-section');
    const draftReplyEl = document.getElementById('draft-reply');
    const approveButton = document.getElementById('approve-button');
    const rejectButton = document.getElementById('reject-button');
    
    const statusMessageEl = document.getElementById('status-message');
    const originalSubmitText = submitButton.textContent;

    let currentTicketId = null;

    // --- Form submission handler ---
    ticketForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        setLoadingState(true);
        hideStatusMessage();
        responseSection.classList.add('hidden');

        currentTicketId = Math.floor(Math.random() * 100000);

        try {
            const response = await fetch('/api/tickets', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: currentTicketId,
                    title: titleInput.value,
                    description: descriptionInput.value,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown server error.' }));
                throw new Error(errorData.detail || 'Network response was not ok.');
            }

            const data = await response.json();
            draftReplyEl.textContent = data.draft_reply;
            responseSection.classList.remove('hidden');

        } catch (error) {
            console.error('Error submitting ticket:', error);
            showStatusMessage(`Error: ${error.message}`, 'error');
        } finally {
            setLoadingState(false);
        }
    });

    // --- Approve button handler ---
    approveButton.addEventListener('click', async () => {
        if (!currentTicketId) return;

        showStatusMessage('Approving and sending to Slack...', 'info');
        approveButton.disabled = true;
        rejectButton.disabled = true;
        
        try {
            const response = await fetch(`/api/approve/${currentTicketId}`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Approval request failed.' }));
                throw new Error(errorData.detail || 'Approval request failed.');
            }

            const data = await response.json();
            if (data.status === 'draft_approved_and_sent') {
                showStatusMessage('Success! Approved and sent to Slack.', 'success');
                responseSection.classList.add('hidden');
                ticketForm.reset();
            } else {
                throw new Error(data.message || 'An unknown approval error occurred.');
            }

        } catch (error) {
            console.error('Error approving draft:', error);
            showStatusMessage(`Error: ${error.message}`, 'error');
        } finally {
            approveButton.disabled = false;
            rejectButton.disabled = false;
        }
    });
    
    // --- Reject button handler ---
    rejectButton.addEventListener('click', () => {
        showStatusMessage('Draft rejected. You can edit and resubmit, or start a new ticket.', 'info');
        responseSection.classList.add('hidden');
        ticketForm.reset();
        titleInput.focus();
        currentTicketId = null;
    });

    // --- Helper Functions ---
    function setLoadingState(isLoading) {
        if (isLoading) {
            submitButton.disabled = true;
            submitButton.innerHTML = `<div class="spinner"></div><span>Generating...</span>`;
        } else {
            submitButton.disabled = false;
            submitButton.innerHTML = originalSubmitText;
        }
    }

    function showStatusMessage(message, type) {
        statusMessageEl.textContent = message;
        statusMessageEl.className = 'status-message'; // Reset classes
        statusMessageEl.classList.add(type, 'visible');
    }

    function hideStatusMessage() {
        statusMessageEl.classList.remove('visible');
    }
});