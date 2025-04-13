// Algorithmic Trading Dashboard JavaScript

// DOM elements
let tradingToggle = document.getElementById('trading-toggle');
let symbolsForm = document.getElementById('symbols-form');
let symbolsInput = document.getElementById('symbols-input');
let accountInfoElement = document.getElementById('account-info');
let portfolioTableBody = document.getElementById('portfolio-table-body');
let tradesTableBody = document.getElementById('trades-table-body');
let statusIndicator = document.getElementById('status-indicator');
let statusText = document.getElementById('status-text');
let alertsContainer = document.getElementById('alerts-container');

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', function() {
    // Enable Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
  
    // Enable Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Set up auto-refresh
    setupAutoRefresh();
});

// Trading toggle functionality
if (tradingToggle) {
    tradingToggle.addEventListener('change', function() {
        const isActive = this.checked;
        const endpoint = isActive ? '/api/start_trading' : '/api/stop_trading';
        
        // Show loading state
        this.disabled = true;
        
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAlert(data.message, 'success');
                updateTradingStatus(isActive);
            } else {
                showAlert(data.message, 'danger');
                // Revert toggle state
                this.checked = !isActive;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('An error occurred: ' + error, 'danger');
            // Revert toggle state
            this.checked = !isActive;
        })
        .finally(() => {
            this.disabled = false;
        });
    });
}

// Symbol management form
if (symbolsForm) {
    symbolsForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const symbolsText = symbolsInput.value.trim();
        if (!symbolsText) {
            showAlert('Please enter at least one symbol', 'warning');
            return;
        }
        
        // Parse symbols
        const symbols = symbolsText.split(',')
            .map(s => s.trim().toUpperCase())
            .filter(s => s.length > 0);
            
        if (symbols.length === 0) {
            showAlert('Please enter valid symbols', 'warning');
            return;
        }
        
        // Update symbols
        updateSymbols(symbols);
    });
}

// Update trading symbols
function updateSymbols(symbols) {
    fetch('/api/update_symbols', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbols: symbols })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(data.message, 'success');
            // Update UI if needed
        } else {
            showAlert(data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred: ' + error, 'danger');
    });
}

// Fetch account info for dashboard
function fetchAccountInfo() {
    if (!accountInfoElement) return;
    
    fetch('/api/account_info')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.account) {
                updateAccountInfo(data.account);
            }
        })
        .catch(error => {
            console.error('Error fetching account info:', error);
        });
}

// Update account info in dashboard
function updateAccountInfo(account) {
    if (!accountInfoElement) return;
    
    let html = `
        <div class="row">
            <div class="col-md-6">
                <p><strong>Portfolio Value:</strong> $${parseFloat(account.portfolio_value).toFixed(2)}</p>
                <p><strong>Cash Balance:</strong> $${parseFloat(account.cash).toFixed(2)}</p>
                <p><strong>Buying Power:</strong> $${parseFloat(account.buying_power).toFixed(2)}</p>
            </div>
            <div class="col-md-6">
                <p><strong>Account Status:</strong> ${account.status}</p>
                <p><strong>Day Trades:</strong> ${account.daytrade_count}</p>
                <p><strong>Last Equity:</strong> $${parseFloat(account.last_equity).toFixed(2)}</p>
            </div>
        </div>
    `;
    
    accountInfoElement.innerHTML = html;
}

// Update trading status indicator
function updateTradingStatus(isActive) {
    if (!statusIndicator || !statusText) return;
    
    if (isActive) {
        statusIndicator.classList.remove('status-inactive');
        statusIndicator.classList.add('status-active');
        statusText.textContent = 'Active';
    } else {
        statusIndicator.classList.remove('status-active');
        statusIndicator.classList.add('status-inactive');
        statusText.textContent = 'Inactive';
    }
}

// Format number as currency
function formatCurrency(value) {
    const num = parseFloat(value);
    return '$' + num.toFixed(2).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

// Format number as percentage
function formatPercentage(value) {
    const num = parseFloat(value);
    return num.toFixed(2) + '%';
}

// Display alert message
function showAlert(message, type = 'info') {
    if (!alertsContainer) return;
    
    const alertId = 'alert-' + Date.now();
    const alertHtml = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertsContainer.innerHTML += alertHtml;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            const bsAlert = new bootstrap.Alert(alertElement);
            bsAlert.close();
        }
    }, 5000);
}

// Setup auto-refresh for dashboard data
function setupAutoRefresh() {
    // Initial fetch
    fetchAccountInfo();
    
    // Refresh every 30 seconds
    setInterval(fetchAccountInfo, 30000);
}
