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
    
    // Initialize the enhanced data tabs if they exist
    initializeEnhancedDataTabs();
    
    // Refresh every 30 seconds
    setInterval(fetchAccountInfo, 30000);
    
    // Refresh enhanced data every 60 seconds
    setInterval(function() {
        if (document.getElementById('news-table-body')) {
            fetchEnhancedData();
        }
    }, 60000);
}

// Initialize the enhanced data tabs
function initializeEnhancedDataTabs() {
    // Check if enhanced data section exists
    if (!document.getElementById('insightsTabs')) return;
    
    // Add event listener to refresh button
    const refreshButton = document.getElementById('refresh-insights');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            fetchEnhancedData();
        });
    }
    
    // Initial data fetch
    fetchEnhancedData();
    
    // Initialize Charts
    initializeAIConfidenceChart();
}

// Fetch all enhanced data
function fetchEnhancedData() {
    fetchNewsSentiment();
    fetchSocialSentiment();
    fetchFundamentalData();
    fetchAIAnalysis();
}

// Fetch news sentiment data
function fetchNewsSentiment() {
    const tableBody = document.getElementById('news-table-body');
    if (!tableBody) return;
    
    // Show loading state
    tableBody.innerHTML = '<tr><td colspan="5" class="text-center">Loading news sentiment data...</td></tr>';
    
    fetch('/api/enhanced_data/news_sentiment')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                displayNewsSentiment(data.data);
            } else {
                tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Error loading news sentiment data</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching news sentiment:', error);
            tableBody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Error loading news sentiment data</td></tr>';
        });
}

// Display news sentiment data
function displayNewsSentiment(data) {
    const tableBody = document.getElementById('news-table-body');
    if (!tableBody) return;
    
    let html = '';
    
    if (data.length === 0) {
        html = '<tr><td colspan="5" class="text-center">No news sentiment data available</td></tr>';
    } else {
        data.forEach(item => {
            // Determine sentiment badge color
            let sentimentBadgeClass = 'bg-secondary';
            if (item.trend === 'positive') {
                sentimentBadgeClass = 'bg-success';
            } else if (item.trend === 'negative') {
                sentimentBadgeClass = 'bg-danger';
            }
            
            // Format headlines
            let headlinesHtml = '<ul class="list-unstyled mb-0">';
            if (item.headlines && item.headlines.length > 0) {
                item.headlines.forEach(headline => {
                    let headlineBadgeClass = 'bg-secondary';
                    if (headline.sentiment === 'positive') {
                        headlineBadgeClass = 'bg-success';
                    } else if (headline.sentiment === 'negative') {
                        headlineBadgeClass = 'bg-danger';
                    }
                    
                    headlinesHtml += `<li class="small mb-1">
                        <span class="badge ${headlineBadgeClass} me-1" style="width: 15px;">&nbsp;</span>
                        ${headline.title}
                    </li>`;
                });
            } else {
                headlinesHtml += '<li class="small text-muted">No recent headlines</li>';
            }
            headlinesHtml += '</ul>';
            
            html += `
                <tr>
                    <td>${item.symbol}</td>
                    <td>
                        <span class="badge ${sentimentBadgeClass}">
                            ${item.sentiment_score.toFixed(2)}
                        </span>
                    </td>
                    <td>
                        <span class="badge ${sentimentBadgeClass}">
                            ${item.trend.toUpperCase()}
                        </span>
                    </td>
                    <td>${headlinesHtml}</td>
                    <td class="small text-muted">${new Date(item.last_updated).toLocaleString()}</td>
                </tr>
            `;
        });
    }
    
    tableBody.innerHTML = html;
}

// Fetch social sentiment data
function fetchSocialSentiment() {
    const tableBody = document.getElementById('social-table-body');
    if (!tableBody) return;
    
    // Show loading state
    tableBody.innerHTML = '<tr><td colspan="6" class="text-center">Loading social sentiment data...</td></tr>';
    
    fetch('/api/enhanced_data/social_sentiment')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                displaySocialSentiment(data.data);
            } else {
                tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading social sentiment data</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching social sentiment:', error);
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading social sentiment data</td></tr>';
        });
}

// Display social sentiment data
function displaySocialSentiment(data) {
    const tableBody = document.getElementById('social-table-body');
    if (!tableBody) return;
    
    let html = '';
    
    if (data.length === 0) {
        html = '<tr><td colspan="6" class="text-center">No social sentiment data available</td></tr>';
    } else {
        data.forEach(item => {
            // Determine sentiment badge colors
            let redditBadgeClass = getSentimentBadgeClass(item.reddit_sentiment);
            let stocktwitsBadgeClass = getSentimentBadgeClass(item.stocktwits_sentiment);
            let combinedBadgeClass = getSentimentBadgeClass(item.combined_score);
            let trendBadgeClass = 'bg-secondary';
            
            if (item.trend === 'positive') {
                trendBadgeClass = 'bg-success';
            } else if (item.trend === 'negative') {
                trendBadgeClass = 'bg-danger';
            }
            
            html += `
                <tr>
                    <td>${item.symbol}</td>
                    <td><span class="badge ${redditBadgeClass}">${item.reddit_sentiment.toFixed(2)}</span></td>
                    <td><span class="badge ${stocktwitsBadgeClass}">${item.stocktwits_sentiment.toFixed(2)}</span></td>
                    <td><span class="badge ${combinedBadgeClass}">${item.combined_score.toFixed(2)}</span></td>
                    <td>${item.mentions}</td>
                    <td><span class="badge ${trendBadgeClass}">${item.trend.toUpperCase()}</span></td>
                </tr>
            `;
        });
    }
    
    tableBody.innerHTML = html;
}

// Helper function to get badge class based on sentiment value
function getSentimentBadgeClass(value) {
    if (value > 0.2) {
        return 'bg-success';
    } else if (value < -0.2) {
        return 'bg-danger';
    } else {
        return 'bg-secondary';
    }
}

// Fetch fundamental data
function fetchFundamentalData() {
    const tableBody = document.getElementById('fundamental-table-body');
    if (!tableBody) return;
    
    // Show loading state
    tableBody.innerHTML = '<tr><td colspan="7" class="text-center">Loading fundamental data...</td></tr>';
    
    fetch('/api/enhanced_data/fundamental')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                displayFundamentalData(data.data);
            } else {
                tableBody.innerHTML = '<tr><td colspan="7" class="text-center text-danger">Error loading fundamental data</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching fundamental data:', error);
            tableBody.innerHTML = '<tr><td colspan="7" class="text-center text-danger">Error loading fundamental data</td></tr>';
        });
}

// Display fundamental data
function displayFundamentalData(data) {
    const tableBody = document.getElementById('fundamental-table-body');
    if (!tableBody) return;
    
    let html = '';
    
    if (data.length === 0) {
        html = '<tr><td colspan="7" class="text-center">No fundamental data available</td></tr>';
    } else {
        data.forEach(item => {
            // Determine company strength badge color
            let strengthBadgeClass = 'bg-secondary';
            if (item.company_strength >= 70) {
                strengthBadgeClass = 'bg-success';
            } else if (item.company_strength >= 50) {
                strengthBadgeClass = 'bg-info';
            } else if (item.company_strength >= 30) {
                strengthBadgeClass = 'bg-warning';
            } else {
                strengthBadgeClass = 'bg-danger';
            }
            
            // Format sector performance value
            let sectorPerformance = (item.sector_performance * 100).toFixed(2) + '%';
            let sectorBadgeClass = item.sector_performance >= 0 ? 'bg-success' : 'bg-danger';
            
            html += `
                <tr>
                    <td>${item.symbol}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar ${strengthBadgeClass}" role="progressbar" 
                                style="width: ${item.company_strength}%;" 
                                aria-valuenow="${item.company_strength}" aria-valuemin="0" aria-valuemax="100">
                                ${item.company_strength}
                            </div>
                        </div>
                    </td>
                    <td>${item.pe_ratio ? item.pe_ratio.toFixed(2) : 'N/A'}</td>
                    <td>${item.profit_margin ? (item.profit_margin * 100).toFixed(2) + '%' : 'N/A'}</td>
                    <td>${item.revenue_growth ? (item.revenue_growth * 100).toFixed(2) + '%' : 'N/A'}</td>
                    <td>${item.debt_to_equity ? item.debt_to_equity.toFixed(2) : 'N/A'}</td>
                    <td>
                        <span class="badge ${sectorBadgeClass}">
                            ${sectorPerformance}
                        </span>
                    </td>
                </tr>
            `;
        });
    }
    
    tableBody.innerHTML = html;
}

// Fetch AI analysis data
function fetchAIAnalysis() {
    const tableBody = document.getElementById('ai-signals-body');
    if (!tableBody) return;
    
    // Show loading state
    tableBody.innerHTML = '<tr><td colspan="6" class="text-center">Loading AI analysis data...</td></tr>';
    
    fetch('/api/enhanced_data/ai_analysis')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                displayAIAnalysis(data.data);
                updateAIConfidenceChart(data.data);
            } else {
                tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading AI analysis data</td></tr>';
            }
        })
        .catch(error => {
            console.error('Error fetching AI analysis:', error);
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading AI analysis data</td></tr>';
        });
}

// Display AI analysis data
function displayAIAnalysis(data) {
    const tableBody = document.getElementById('ai-signals-body');
    if (!tableBody) return;
    
    let html = '';
    
    if (data.length === 0) {
        html = '<tr><td colspan="6" class="text-center">No AI analysis data available</td></tr>';
    } else {
        data.forEach(item => {
            html += `
                <tr>
                    <td>${item.symbol}</td>
                    <td>${getSignalBadge(item.technical.signal, item.technical.action)}</td>
                    <td>${getSignalBadge(item.news.signal, item.news.action)}</td>
                    <td>${getSignalBadge(item.social.signal, item.social.action)}</td>
                    <td>${getSignalBadge(item.fundamental.signal, item.fundamental.action)}</td>
                    <td>${getCombinedSignalBadge(item.combined.signal, item.combined.action, item.combined.confidence)}</td>
                </tr>
            `;
        });
    }
    
    tableBody.innerHTML = html;
}

// Helper function to generate signal badge HTML
function getSignalBadge(signal, action) {
    let badgeClass = 'bg-secondary';
    if (action === 'buy') {
        badgeClass = 'bg-success';
    } else if (action === 'sell') {
        badgeClass = 'bg-danger';
    }
    
    return `<span class="badge ${badgeClass}">${signal.toFixed(2)}</span>`;
}

// Helper function to generate combined signal badge HTML
function getCombinedSignalBadge(signal, action, confidence) {
    let badgeClass = 'bg-secondary';
    if (action === 'buy') {
        badgeClass = 'bg-success';
    } else if (action === 'sell') {
        badgeClass = 'bg-danger';
    }
    
    let confidenceText = '';
    if (confidence > 0.7) {
        confidenceText = 'High';
    } else if (confidence > 0.4) {
        confidenceText = 'Medium';
    } else {
        confidenceText = 'Low';
    }
    
    return `
        <span class="badge ${badgeClass} me-1">${signal.toFixed(2)}</span>
        <span class="badge bg-info">${confidenceText}</span>
    `;
}

// AI Confidence Chart
let aiConfidenceChart = null;

// Initialize AI Confidence Chart
function initializeAIConfidenceChart() {
    const canvas = document.getElementById('aiConfidenceChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    aiConfidenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'AI Prediction Confidence',
                data: [],
                backgroundColor: [],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'Confidence'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Symbols'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const confidence = context.raw;
                            let confidenceText = 'Low';
                            if (confidence > 0.7) {
                                confidenceText = 'High';
                            } else if (confidence > 0.4) {
                                confidenceText = 'Medium';
                            }
                            return `Confidence: ${(confidence * 100).toFixed(1)}% (${confidenceText})`;
                        }
                    }
                }
            }
        }
    });
}

// Update AI Confidence Chart
function updateAIConfidenceChart(data) {
    if (!aiConfidenceChart) return;
    
    const labels = [];
    const confidenceData = [];
    const backgroundColors = [];
    
    data.forEach(item => {
        labels.push(item.symbol);
        confidenceData.push(item.combined.confidence);
        
        // Determine bar color based on action
        let color = '#6c757d'; // Default gray for hold
        if (item.combined.action === 'buy') {
            color = '#198754'; // Bootstrap success color
        } else if (item.combined.action === 'sell') {
            color = '#dc3545'; // Bootstrap danger color
        }
        
        backgroundColors.push(color);
    });
    
    aiConfidenceChart.data.labels = labels;
    aiConfidenceChart.data.datasets[0].data = confidenceData;
    aiConfidenceChart.data.datasets[0].backgroundColor = backgroundColors;
    
    aiConfidenceChart.update();
}
