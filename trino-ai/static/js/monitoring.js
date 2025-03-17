// Monitoring Dashboard JavaScript

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Load initial dashboard data
    refreshDashboard();
    
    // Set up automatic refresh every 30 seconds
    setInterval(refreshDashboard, 30000);
    
    // Set up event handlers for filtering
    setupFilterHandlers();
});

function refreshDashboard() {
    // Fetch dashboard data from API
    fetch('/api/monitoring/dashboard')
        .then(response => response.json())
        .then(data => {
            updateQueryTable(data.queries);
            updateErrorLog(data.errors);
            updateSystemHealth(data.health);
            updateAgentActivity(data.agent_activity);
            updateCharts(data);
        })
        .catch(error => {
            console.error('Error refreshing dashboard:', error);
            showErrorMessage('Failed to load dashboard data. Please try again later.');
        });
}

function updateQueryTable(queries) {
    const tableBody = document.getElementById('recent-queries-body');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    queries.forEach(query => {
        const row = document.createElement('tr');
        
        // Add query ID cell
        const idCell = document.createElement('td');
        idCell.textContent = query.query_id;
        row.appendChild(idCell);
        
        // Add timestamp cell
        const timestampCell = document.createElement('td');
        timestampCell.textContent = formatDateTime(query.timestamp);
        row.appendChild(timestampCell);
        
        // Add query text cell
        const queryTextCell = document.createElement('td');
        queryTextCell.textContent = query.natural_language;
        queryTextCell.className = 'query-text';
        row.appendChild(queryTextCell);
        
        // Add status cell
        const statusCell = document.createElement('td');
        statusCell.innerHTML = `<span class="status-badge status-${query.status}">${query.status}</span>`;
        row.appendChild(statusCell);
        
        // Add execution time cell
        const timeCell = document.createElement('td');
        timeCell.textContent = `${query.execution_time_ms}ms`;
        row.appendChild(timeCell);
        
        // Add action cell
        const actionCell = document.createElement('td');
        actionCell.innerHTML = `
            <button class="btn btn-sm btn-outline-primary view-sql-btn" data-query-id="${query.query_id}">
                View SQL
            </button>
        `;
        row.appendChild(actionCell);
        
        tableBody.appendChild(row);
    });
    
    // Add event listeners to view SQL buttons
    document.querySelectorAll('.view-sql-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const queryId = this.getAttribute('data-query-id');
            viewSqlDetails(queryId);
        });
    });
}

function updateErrorLog(errors) {
    const errorContainer = document.getElementById('error-log-container');
    if (!errorContainer) return;
    
    errorContainer.innerHTML = '';
    
    if (errors.length === 0) {
        errorContainer.innerHTML = '<div class="alert alert-success">No errors reported.</div>';
        return;
    }
    
    errors.forEach(error => {
        const errorCard = document.createElement('div');
        errorCard.className = 'card error-card mb-3';
        
        errorCard.innerHTML = `
            <div class="card-header bg-danger text-white">
                <strong>${error.error_type}</strong> - ${formatDateTime(error.timestamp)}
            </div>
            <div class="card-body">
                <p class="card-text">${error.message}</p>
                <p class="text-muted">Source: ${error.source}, Query ID: ${error.query_id}</p>
                ${error.stack_trace ? 
                  `<button class="btn btn-sm btn-outline-secondary toggle-stack-trace">
                     Show Stack Trace
                   </button>
                   <pre class="stack-trace mt-2" style="display: none;">${error.stack_trace}</pre>` 
                  : ''}
            </div>
        `;
        
        errorContainer.appendChild(errorCard);
    });
    
    // Add event listeners to toggle stack trace buttons
    document.querySelectorAll('.toggle-stack-trace').forEach(btn => {
        btn.addEventListener('click', function() {
            const stackTrace = this.nextElementSibling;
            if (stackTrace.style.display === 'none') {
                stackTrace.style.display = 'block';
                this.textContent = 'Hide Stack Trace';
            } else {
                stackTrace.style.display = 'none';
                this.textContent = 'Show Stack Trace';
            }
        });
    });
}

function updateSystemHealth(health) {
    const healthContainer = document.getElementById('system-health-container');
    if (!healthContainer) return;
    
    const statusClass = health.status === 'healthy' ? 'success' : 'warning';
    
    healthContainer.innerHTML = `
        <div class="card">
            <div class="card-header bg-${statusClass} text-white">
                System Health: ${health.status.toUpperCase()}
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Uptime:</strong> ${formatDuration(health.uptime_seconds)}</p>
                        <p><strong>Started:</strong> ${formatDateTime(health.startup_time)}</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Current Time:</strong> ${formatDateTime(health.current_time)}</p>
                        ${health.last_critical_error ? 
                          `<p><strong>Last Critical Error:</strong> ${formatDateTime(health.last_critical_error)}</p>` 
                          : ''}
                    </div>
                </div>
            </div>
        </div>
    `;
}

function updateAgentActivity(activities) {
    const activityContainer = document.getElementById('agent-activity-container');
    if (!activityContainer) return;
    
    activityContainer.innerHTML = '';
    
    // Group activities by agent
    const groupedActivities = {};
    
    activities.forEach(activity => {
        if (!groupedActivities[activity.agent]) {
            groupedActivities[activity.agent] = [];
        }
        groupedActivities[activity.agent].push(activity);
    });
    
    // Create a tab for each agent
    const tabList = document.createElement('ul');
    tabList.className = 'nav nav-tabs';
    
    const tabContent = document.createElement('div');
    tabContent.className = 'tab-content mt-2';
    
    let isFirst = true;
    
    Object.keys(groupedActivities).forEach(agent => {
        // Create tab
        const tabItem = document.createElement('li');
        tabItem.className = 'nav-item';
        
        const tabLink = document.createElement('a');
        tabLink.className = `nav-link ${isFirst ? 'active' : ''}`;
        tabLink.setAttribute('data-toggle', 'tab');
        tabLink.setAttribute('href', `#${agent.replace(/\s+/g, '-')}-tab`);
        tabLink.textContent = agent;
        
        tabItem.appendChild(tabLink);
        tabList.appendChild(tabItem);
        
        // Create tab content
        const tabPane = document.createElement('div');
        tabPane.className = `tab-pane fade ${isFirst ? 'show active' : ''}`;
        tabPane.id = `${agent.replace(/\s+/g, '-')}-tab`;
        
        const table = document.createElement('table');
        table.className = 'table table-sm';
        
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Action</th>
                    <th>Duration</th>
                    <th>Query ID</th>
                </tr>
            </thead>
            <tbody>
                ${groupedActivities[agent].map(activity => `
                    <tr>
                        <td>${formatDateTime(activity.timestamp)}</td>
                        <td>${activity.action}</td>
                        <td>${activity.duration_ms}ms</td>
                        <td>${activity.query_id}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        
        tabPane.appendChild(table);
        tabContent.appendChild(tabPane);
        
        isFirst = false;
    });
    
    activityContainer.appendChild(tabList);
    activityContainer.appendChild(tabContent);
}

function updateCharts(data) {
    // Update charts with the data
    // This would use a charting library like Chart.js
    
    // Example:
    // updateQueryTimeChart(data.queries);
    // updateErrorRateChart(data.errors);
    // etc.
}

function viewSqlDetails(queryId) {
    // Fetch query details and show SQL in a modal
    fetch(`/api/monitoring/queries/${queryId}`)
        .then(response => response.json())
        .then(data => {
            showSqlModal(data);
        })
        .catch(error => {
            console.error('Error fetching query details:', error);
            showErrorMessage('Failed to load query details.');
        });
}

function showSqlModal(queryData) {
    // Create and show a modal with SQL details
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'sqlModal';
    modal.setAttribute('tabindex', '-1');
    
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">SQL Query Details</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="card mb-3">
                        <div class="card-header">Natural Language Query</div>
                        <div class="card-body">
                            <p>${queryData.natural_language}</p>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">Generated SQL</div>
                        <div class="card-body">
                            <pre><code class="sql">${queryData.generated_sql}</code></pre>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Initialize and show the modal
    $('#sqlModal').modal('show');
    
    // Handle cleanup when modal is closed
    $('#sqlModal').on('hidden.bs.modal', function () {
        $(this).remove();
    });
}

function setupFilterHandlers() {
    // Set up handlers for any filter controls
    const statusFilter = document.getElementById('status-filter');
    if (statusFilter) {
        statusFilter.addEventListener('change', function() {
            const status = this.value;
            // Apply filter to tables or make API call with filter
        });
    }
    
    const dateRangeFilter = document.getElementById('date-range-filter');
    if (dateRangeFilter) {
        dateRangeFilter.addEventListener('change', function() {
            const range = this.value;
            // Apply date range filter
        });
    }
}

function showErrorMessage(message) {
    // Show an error message to the user
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.setAttribute('role', 'alert');
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        $(alertDiv).alert('close');
    }, 5000);
}

// Helper functions
function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

function formatDuration(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    const parts = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);
    
    return parts.join(' ');
} 