/**
 * Feature Store Data Viewer Panel
 * Displays actual feature store data from parquet files
 */
class FeatureDataPanel {
    constructor() {
        this.panelId = 'featuredata';
        this.container = null;
        this.featureData = null;
        this.currentPage = 1;
        this.pageSize = 50;
        this.isLoading = false;
        this.error = null;
    }

    /**
     * Initialize the feature data panel
     * This can be called standalone or integrated into the featurestore panel
     */
    init(containerId = null) {
        if (containerId) {
            this.container = document.getElementById(containerId);
        } else {
            const panel = document.getElementById(`panel-${this.panelId}`);
            if (panel) {
                this.container = panel.querySelector('.panel-content');
            }
        }

        if (!this.container) {
            console.warn('Feature Data panel container not found');
            return;
        }

        this.render();
    }

    /**
     * Render the feature data panel content
     */
    render() {
        this.container.innerHTML = '';

        // Overview section
        const overviewSection = this.renderOverviewSection();
        this.container.appendChild(overviewSection);

        // Predictor features section
        const predictorSection = this.renderPredictorSection();
        this.container.appendChild(predictorSection);

        // Target features section
        const targetSection = this.renderTargetSection();
        this.container.appendChild(targetSection);
    }

    /**
     * Render the overview section
     * @returns {HTMLElement} - Overview section element
     */
    renderOverviewSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📊 Feature Store Data Viewer</h3>
                <button class="btn btn-outline" id="refresh-featuredata-btn">
                    🔄 Refresh
                </button>
            </div>
            <div class="card-body">
                <p class="text-muted">
                    View the actual feature data stored in the feature store. 
                    Data is loaded from parquet files and includes entity keys and event timestamps.
                </p>
                <div class="ttl-info mt-md" id="ttl-info-container">
                    <!-- TTL info will be rendered here -->
                </div>
            </div>
        `;

        // Add refresh button handler
        setTimeout(() => {
            const refreshBtn = section.querySelector('#refresh-featuredata-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.loadFeatureData(false));
            }
        }, 0);

        return section;
    }

    /**
     * Render the predictor features section
     * @returns {HTMLElement} - Predictor section element
     */
    renderPredictorSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'predictor-data-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🎯 Predictor Features</h3>
                <span class="badge badge-primary">predictors_df_feature_view</span>
            </div>
            <div class="card-body" id="predictor-data-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading predictor features...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the target features section
     * @returns {HTMLElement} - Target section element
     */
    renderTargetSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'target-data-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🎲 Target Features</h3>
                <span class="badge badge-warning">ptarget_df_feature_view</span>
            </div>
            <div class="card-body" id="target-data-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading target features...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Load feature data from the API
     * @param {boolean} useCache - Whether to use cached data
     */
    async loadFeatureData(useCache = true) {
        const predictorContainer = document.getElementById('predictor-data-content');
        const targetContainer = document.getElementById('target-data-content');
        const ttlContainer = document.getElementById('ttl-info-container');

        if (!predictorContainer || !targetContainer) return;

        this.isLoading = true;
        this.error = null;

        // Show loading states using helper function if available
        const loadingHtml = window.renderLoadingState 
            ? window.renderLoadingState('Loading feature data...')
            : `<div class="loading"><div class="loading-spinner"></div><span>Loading...</span></div>`;
        
        predictorContainer.innerHTML = loadingHtml;
        targetContainer.innerHTML = loadingHtml;

        try {
            this.featureData = await dataLoader.fetchFeatureStoreData(this.currentPage, this.pageSize, useCache);

            // Render TTL info
            if (ttlContainer && this.featureData.ttl) {
                this.renderTTLInfo(ttlContainer);
            }

            // Render predictor data
            if (this.featureData.predictors) {
                this.renderFeatureTable(predictorContainer, this.featureData.predictors, 'predictor');
            } else {
                predictorContainer.innerHTML = '<div class="no-data-message">No predictor data available</div>';
            }

            // Render target data
            if (this.featureData.targets) {
                this.renderFeatureTable(targetContainer, this.featureData.targets, 'target');
            } else {
                targetContainer.innerHTML = '<div class="no-data-message">No target data available</div>';
            }
        } catch (error) {
            this.error = error;
            this.renderError(predictorContainer, error);
            this.renderError(targetContainer, error);
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Render TTL information
     * @param {HTMLElement} container - Container element
     */
    renderTTLInfo(container) {
        const ttl = this.featureData.ttl;
        const days = Math.floor(ttl / 86400);
        
        container.innerHTML = `
            <div class="ttl-badge">
                <span>⏱️ TTL:</span>
                <span>${days} days (${ttl.toLocaleString()} seconds)</span>
            </div>
            <p class="text-muted mt-sm" style="font-size: 0.85rem;">
                Features older than the TTL will not be served from the online store.
            </p>
        `;
    }

    /**
     * Render a feature data table
     * @param {HTMLElement} container - Container element
     * @param {Object} data - Feature data object
     * @param {string} type - Type of features ('predictor' or 'target')
     */
    renderFeatureTable(container, data, type) {
        if (!data || !data.data || data.data.length === 0) {
            container.innerHTML = '<div class="no-data-message">No data available</div>';
            return;
        }

        const columns = data.columns || [];
        const rows = data.data || [];

        // Identify special columns
        const entityColumn = columns.find(c => c.toLowerCase().includes('patient_id') || c === 'patient_id');
        const timestampColumn = columns.find(c => c.toLowerCase().includes('timestamp') || c.includes('event_timestamp'));

        // Build table header
        const headerHtml = columns.map(col => {
            let className = '';
            if (col === entityColumn) className = 'entity-key-header';
            if (col === timestampColumn) className = 'timestamp-header';
            return `<th class="${className}">${col}</th>`;
        }).join('');

        // Build table rows
        const rowsHtml = rows.map(row => {
            const cellsHtml = columns.map(col => {
                let value = row[col];
                let className = '';
                
                if (col === entityColumn) {
                    className = 'entity-key-cell';
                } else if (col === timestampColumn) {
                    className = 'timestamp-cell';
                    // Format timestamp if it's a string
                    if (typeof value === 'string' && value.includes('T')) {
                        value = this.formatTimestamp(value);
                    }
                } else if (typeof value === 'number') {
                    // Format numbers
                    value = Number.isInteger(value) ? value : value.toFixed(4);
                }
                
                return `<td class="${className}">${value !== null && value !== undefined ? value : 'N/A'}</td>`;
            }).join('');
            
            return `<tr>${cellsHtml}</tr>`;
        }).join('');

        // Build pagination
        const paginationHtml = this.renderPagination(data, type);

        container.innerHTML = `
            <div class="feature-data-section">
                <div class="table-stats">
                    <span class="stats-info">
                        Showing ${rows.length} of ${data.total.toLocaleString()} rows
                    </span>
                </div>
                <div class="feature-data-table-container">
                    <table class="feature-data-table">
                        <thead>
                            <tr>${headerHtml}</tr>
                        </thead>
                        <tbody>
                            ${rowsHtml}
                        </tbody>
                    </table>
                </div>
                ${paginationHtml}
            </div>
        `;

        // Add pagination event listeners
        this.setupPaginationListeners(container, type);
    }

    /**
     * Render pagination controls
     * @param {Object} data - Feature data object
     * @param {string} type - Type of features
     * @returns {string} - Pagination HTML
     */
    renderPagination(data, type) {
        const { page, totalPages } = data;
        
        if (totalPages <= 1) return '';

        const prevDisabled = page <= 1 ? 'disabled' : '';
        const nextDisabled = page >= totalPages ? 'disabled' : '';

        // Generate page numbers
        let pageNumbers = [];
        const maxVisible = 5;
        let start = Math.max(1, page - Math.floor(maxVisible / 2));
        let end = Math.min(totalPages, start + maxVisible - 1);
        
        if (end - start < maxVisible - 1) {
            start = Math.max(1, end - maxVisible + 1);
        }

        for (let i = start; i <= end; i++) {
            const activeClass = i === page ? 'active' : '';
            pageNumbers.push(`<button class="pagination-btn ${activeClass}" data-page="${i}" data-type="${type}">${i}</button>`);
        }

        return `
            <div class="pagination-wrapper">
                <div class="pagination">
                    <button class="pagination-btn" data-page="${page - 1}" data-type="${type}" ${prevDisabled}>← Prev</button>
                    ${start > 1 ? '<span class="pagination-ellipsis">...</span>' : ''}
                    ${pageNumbers.join('')}
                    ${end < totalPages ? '<span class="pagination-ellipsis">...</span>' : ''}
                    <button class="pagination-btn" data-page="${page + 1}" data-type="${type}" ${nextDisabled}>Next →</button>
                </div>
                <div class="pagination-info">
                    Page ${page} of ${totalPages}
                </div>
            </div>
        `;
    }

    /**
     * Setup pagination event listeners
     * @param {HTMLElement} container - Container element
     */
    setupPaginationListeners(container) {
        const buttons = container.querySelectorAll('.pagination-btn[data-page]');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = parseInt(e.target.getAttribute('data-page'));
                if (!isNaN(page) && page >= 1) {
                    this.handlePageChange(page);
                }
            });
        });
    }

    /**
     * Handle page change
     * @param {number} page - New page number
     */
    async handlePageChange(page) {
        this.currentPage = page;
        await this.loadFeatureData(false);
    }

    /**
     * Format timestamp for display
     * @param {string} timestamp - ISO timestamp string
     * @returns {string} - Formatted timestamp
     */
    formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return timestamp;
        }
    }

    /**
     * Render error state
     * @param {HTMLElement} container - Container element
     * @param {Error} error - Error object
     */
    renderError(container, error) {
        const errorMessage = error instanceof DataLoaderError 
            ? error.getUserMessage() 
            : (error.message || 'Failed to load feature data');

        container.innerHTML = `
            <div class="retry-section">
                <div class="error-message">
                    <span>⚠️</span>
                    <span>${errorMessage}</span>
                </div>
                <button class="btn btn-retry retry-featuredata-btn">
                    🔄 Try Again
                </button>
            </div>
        `;

        const retryBtn = container.querySelector('.retry-featuredata-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.loadFeatureData(false));
        }
    }

    /**
     * Called when the panel becomes visible
     */
    onActivate() {
        console.log('Feature Data panel activated');
        // Load feature data when panel is activated
        if (!this.featureData && !this.isLoading) {
            this.loadFeatureData();
        }
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('Feature Data panel deactivated');
    }
}

// Create global instance
const featureDataPanel = new FeatureDataPanel();

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.FeatureDataPanel = FeatureDataPanel;
    window.featureDataPanel = featureDataPanel;
}
