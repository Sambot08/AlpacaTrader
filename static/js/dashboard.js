document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts and data
    initializeDashboard();
    
    // Set up event listeners
    setupEventListeners();
});

// Chart objects
let portfolioChart = null;

// Initialize the dashboard
function initializeDashboard() {
    // Load initial data
    loadPerformanceData();
    loadRecentTrades();
    
    // Refresh data every 5 minutes
    setInterval(function() {
        loadPerformanceData();
        loadRecentTrades();
    }, 300000); // 5 minutes
}

// Load performance data and update charts/metrics
function loadPerformanceData(days = 30) {
    fetch(`/api/performance_data?days=${days}`)
        .then(response => response.json())
        .then(response => {
            if (response.success) {
                updatePerformanceCharts(response.data);
                updateMetrics(response.data);
            } else {
                console.error('Error loading performance data:', response.message);
            }
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
}

// Load recent trades
function loadRecentTrades() {
    fetch('/api/recent_trades?limit=10')
        .then(response => response.json())
        .then(response => {
            if (response.success) {
                updateTradesTable(response.trades);
            } else {
                console.error('Error loading recent trades:', response.message);
            }
        })
        .catch(error => {
            console.error('Error fetching recent trades:', error);
        });
}

// Update the performance charts
function updatePerformanceCharts(data) {
    const ctx = document.getElementById('portfolioChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (portfolioChart) {
        portfolioChart.destroy();
    }
    
    // Create new chart
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Portfolio Value ($)',
                    data: data.portfolio_values,
                    borderColor: 'rgba(78, 115, 223, 1)',
                    backgroundColor: 'rgba(78, 115, 223, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: 'rgba(78, 115, 223, 1)',
                    pointBorderColor: 'rgba(78, 115, 223, 1)',
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: 'rgba(78, 115, 223, 1)',
                    pointHoverBorderColor: 'rgba(78, 115, 223, 1)',
                    pointHitRadius: 10,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + context.parsed.y.toLocaleString(undefined, {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2
                                });
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
    
    // Update summary metrics if we have data
    if (data.portfolio_values.length > 0) {
        const latestValue = data.portfolio_values[data.portfolio_values.length - 1];
        document.getElementById('portfolio-value').innerText = '$' + latestValue.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
        
        // Calculate daily return
        if (data.returns.length > 0) {
            const latestReturn = data.returns[data.returns.length - 1];
            const returnEl = document.getElementById('daily-return');
            returnEl.innerText = latestReturn.toFixed(2) + '%';
            
            // Color based on positive/negative
            if (latestReturn > 0) {
                returnEl.classList.add('text-success');
                returnEl.classList.remove('text-danger');
            } else if (latestReturn < 0) {
                returnEl.classList.add('text-danger');
                returnEl.classList.remove('text-success');
            }
        }
    }
}

// Update metrics display
function updateMetrics(data) {
    // These would typically come from performance metrics API
    // For now, we'll calculate some based on the chart data
    
    // Calculate a simple Sharpe ratio (estimate)
    if (data.returns && data.returns.length > 0) {
        const returns = data.returns;
        const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const stdDev = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
        const sharpeRatio = (stdDev === 0) ? 0 : (avgReturn / stdDev) * Math.sqrt(252); // Annualized
        
        document.getElementById('sharpe-ratio').innerText = sharpeRatio.toFixed(2);
        
        // Calculate simple win rate (positive return days)
        const winCount = returns.filter(ret => ret > 0).length;
        const winRate = (winCount / returns.length) * 100;
        document.getElementById('win-rate').innerText = winRate.toFixed(2) + '%';
        
        // Calculate max drawdown
        let peak = data.portfolio_values[0];
        let maxDrawdown = 0;
        
        for (let i = 1; i < data.portfolio_values.length; i++) {
            const value = data.portfolio_values[i];
            if (value > peak) {
                peak = value;
            } else {
                const drawdown = (peak - value) / peak * 100;
                if (drawdown > maxDrawdown) {
                    maxDrawdown = drawdown;
                }
            }
        }
        
        document.getElementById('max-drawdown').innerText = maxDrawdown.toFixed(2) + '%';
        
        // Profit factor (total gain / total loss)
        const gains = returns.filter(ret => ret > 0).reduce((sum, ret) => sum + ret, 0);
        const losses = Math.abs(returns.filter(ret => ret < 0).reduce((sum, ret) => sum + ret, 0));
        const profitFactor = losses === 0 ? gains : gains / losses;
        document.getElementById('profit-factor').innerText = profitFactor.toFixed(2);
        
        // Average trade return
        document.getElementById('avg-trade-return').innerText = avgReturn.toFixed(2) + '%';
        
        // Total trades
        document.getElementById('total-trades').innerText = returns.length;
    }
}

// Update the trades table
function updateTradesTable(trades) {
    const tableBody = document.getElementById('trades-table-body');
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    // Add new rows
    trades.forEach(trade => {
        const row = document.createElement('tr');
        
        // Format the row
        row.innerHTML = `
            <td>${trade.symbol}</td>
            <td>
                <span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">
                    ${trade.action}
                </span>
            </td>
            <td>${trade.quantity}</td>
            <td>$${trade.price.toFixed(2)}</td>
            <td>$${(trade.price * trade.quantity).toFixed(2)}</td>
            <td>${trade.timestamp}</td>
            <td><span class="badge bg-info">${trade.status || 'FILLED'}</span></td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Time period buttons
    const periodButtons = document.querySelectorAll('[data-period]');
    periodButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active state
            periodButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Load data for selected period
            const period = parseInt(this.getAttribute('data-period'));
            loadPerformanceData(period);
        });
    });
    
    // Refresh trades button
    const refreshButton = document.getElementById('refresh-trades');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            loadRecentTrades();
        });
    }
    
    // Start/stop strategy buttons
    document.addEventListener('click', function(e) {
        if (e.target.closest('.start-strategy')) {
            const button = e.target.closest('.start-strategy');
            const strategyId = button.getAttribute('data-strategy-id');
            startStrategy(strategyId);
        } else if (e.target.closest('.stop-strategy')) {
            const button = e.target.closest('.stop-strategy');
            const strategyId = button.getAttribute('data-strategy-id');
            stopStrategy(strategyId);
        }
    });
}

// Start a trading strategy
function startStrategy(strategyId) {
    fetch('/api/start_trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ strategy_id: strategyId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reload the page to reflect changes
            window.location.reload();
        } else {
            alert('Error starting strategy: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error starting strategy');
    });
}

// Stop a trading strategy
function stopStrategy(strategyId) {
    fetch('/api/stop_trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ strategy_id: strategyId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Reload the page to reflect changes
            window.location.reload();
        } else {
            alert('Error stopping strategy: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error stopping strategy');
    });
}
