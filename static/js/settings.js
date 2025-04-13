document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    setupStrategyFormEvents();
    setupAPIFormEvents();
    setupStrategyListEvents();
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
    const apiForm = document.getElementById('api-form');
    
    if (apiForm) {
        apiForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const apiKey = document.getElementById('api_key').value;
            const apiSecret = document.getElementById('api_secret').value;
            const paperTrading = document.getElementById('paper_trading').checked;
            
            if (!apiKey || !apiSecret) {
                alert('Please enter your Alpaca API key and secret.');
                return;
            }
            
            // In a real application, you would save these credentials securely
            // Here we'll just show a confirmation message
            alert('API settings saved successfully!');
            
            // Clear the form
            apiForm.reset();
        });
    }
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
