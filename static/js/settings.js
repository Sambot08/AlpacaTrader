document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupStrategyFormEvents();
    setupAPIFormEvents();
    setupStrategyListEvents();
    setupTestConnectionButton();
});

function setupStrategyFormEvents() {
    const strategyForm = document.getElementById('strategy-form');
    
    if (strategyForm) {
        strategyForm.addEventListener('submit', function(e) {
            // Form is submitted normally - handled by Flask
        });
    }
}

function setupAPIFormEvents() {
    // This is handled by the main form submit now
}

function setupStrategyListEvents() {
    // Edit strategy buttons
    const editButtons = document.querySelectorAll('.edit-strategy');
    editButtons.forEach(button => {
        button.addEventListener('click', function() {
            const listItem = this.closest('.strategy-item');
            const strategyId = listItem.getAttribute('data-strategy-id');
            const strategyName = listItem.getAttribute('data-strategy-name');
            const strategySymbols = listItem.getAttribute('data-strategy-symbols');
            const strategyModel = listItem.getAttribute('data-strategy-model');
            const strategyRisk = listItem.getAttribute('data-strategy-risk');
            
            // Populate the form with this strategy's data
            document.getElementById('strategy_name').value = strategyName;
            document.getElementById('stock_symbols').value = strategySymbols;
            document.getElementById('ml_model_type').value = strategyModel;
            document.getElementById('risk_level').value = strategyRisk;
            
            // Scroll to the form
            document.getElementById('strategy-form').scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Delete strategy buttons
    const deleteButtons = document.querySelectorAll('.delete-strategy');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function() {
            const listItem = this.closest('.strategy-item');
            const strategyId = listItem.getAttribute('data-strategy-id');
            const strategyName = listItem.getAttribute('data-strategy-name');
            
            if (confirm(`Are you sure you want to delete the strategy "${strategyName}"?`)) {
                // In a real application, you would send a delete request to the server
                // For this example, we'll just remove the item from the DOM
                listItem.remove();
                
                // If the list is now empty, show a message
                const strategyList = document.querySelector('.strategy-list');
                if (strategyList.children.length === 0) {
                    const noStrategiesMsg = document.createElement('div');
                    noStrategiesMsg.className = 'text-center py-3';
                    noStrategiesMsg.innerHTML = '<p class="text-muted">No strategies created yet.</p>';
                    strategyList.appendChild(noStrategiesMsg);
                }
            }
        });
    });
}

function setupTestConnectionButton() {
    const testButton = document.getElementById('test_connection');
    const connectionStatus = document.getElementById('connection_status');
    
    if (testButton && connectionStatus) {
        testButton.addEventListener('click', function() {
            const apiKey = document.getElementById('api_key').value.trim();
            const apiSecret = document.getElementById('api_secret').value.trim();
            const apiBaseUrl = document.getElementById('api_base_url').value;
            
            if (!apiKey || !apiSecret) {
                connectionStatus.className = 'alert alert-danger';
                connectionStatus.textContent = 'Please enter your API key and secret first.';
                connectionStatus.classList.remove('d-none');
                return;
            }
            
            // Show checking status
            connectionStatus.className = 'alert alert-info';
            connectionStatus.textContent = 'Testing connection to Alpaca API...';
            connectionStatus.classList.remove('d-none');
            
            // Send request to test the connection
            fetch('/api/test_connection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    api_secret: apiSecret,
                    api_base_url: apiBaseUrl
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    connectionStatus.className = 'alert alert-success';
                    connectionStatus.innerHTML = `<strong>Success!</strong> Connected to Alpaca API.<br>Account: ${data.account.account_number}<br>Cash Balance: $${parseFloat(data.account.cash).toFixed(2)}<br>Portfolio Value: $${parseFloat(data.account.portfolio_value).toFixed(2)}`;
                } else {
                    connectionStatus.className = 'alert alert-danger';
                    connectionStatus.textContent = `Connection failed: ${data.message}`;
                }
            })
            .catch(error => {
                connectionStatus.className = 'alert alert-danger';
                connectionStatus.textContent = `Error testing connection: ${error.message}`;
            });
        });
    }
}
