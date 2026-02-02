/**
 * Dataset Viewer Panel
 * Displays the diabetes.csv dataset with pagination and filtering
 */
class DatasetPanel {
    constructor() {
        this.panelId = 'dataset';
        this.container = null;
        this.dataTable = null;
        this.datasetInfo = null;
        this.dvcInfo = null;
        this.currentPage = 1;
        this.pageSize = 100;
        this.isLoading = false;
        this.error = null;
    }

    /**
     * Initialize the dataset panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('Dataset panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('Dataset panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the dataset panel content
     */
    render() {
        this.container.innerHTML = '';

        // Header section with metadata
        const headerSection = this.renderHeaderSection();
        this.container.appendChild(headerSection);

        // Data table section
        const tableSection = this.renderTableSection();
        this.container.appendChild(tableSection);
    }

    /**
     * Render the header section with dataset info
     * @returns {HTMLElement} - Header section element
     */
    renderHeaderSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'dataset-header-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📋 Dataset Viewer</h3>
                <button class="btn btn-outline" id="refresh-dataset-btn">
                    🔄 Refresh
                </button>
            </div>
            <div class="card-body">
                <div class="dataset-header" id="dataset-meta-container">
                    <div class="loading">
                        <div class="loading-spinner"></div>
                        <span>Loading dataset info...</span>
                    </div>
                </div>
            </div>
        `;

        // Add refresh button handler
        setTimeout(() => {
            const refreshBtn = section.querySelector('#refresh-dataset-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.loadDataset(false));
            }
        }, 0);

        return section;
    }

    /**
     * Render the table section
     * @returns {HTMLElement} - Table section element
     */
    renderTableSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'dataset-table-section';

        section.innerHTML = `
            <div class="card-body" id="dataset-table-container">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading dataset...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Load dataset from the API
     * @param {boolean} useCache - Whether to use cached data
     */
    async loadDataset(useCache = true) {
        const metaContainer = document.getElementById('dataset-meta-container');
        const tableContainer = document.getElementById('dataset-table-container');
        
        if (!metaContainer || !tableContainer) return;

        this.isLoading = true;
        this.error = null;

        // Show loading states using helper function if available
        const loadingHtml = window.renderLoadingState 
            ? window.renderLoadingState('Loading dataset...')
            : `<div class="loading"><div class="loading-spinner"></div><span>Loading...</span></div>`;
        
        metaContainer.innerHTML = loadingHtml;
        tableContainer.innerHTML = loadingHtml;

        try {
            // Load dataset and DVC info in parallel
            const [datasetResponse, dvcResponse] = await Promise.all([
                dataLoader.fetchDataset(this.currentPage, this.pageSize, useCache),
                dataLoader.fetchDVCInfo(useCache)
            ]);

            this.datasetInfo = datasetResponse;
            this.dvcInfo = dvcResponse;

            this.renderDatasetMeta(metaContainer);
            this.renderDataTable(tableContainer);
        } catch (error) {
            this.error = error;
            this.renderError(metaContainer, tableContainer, error);
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Render dataset metadata
     * @param {HTMLElement} container - Container element
     */
    renderDatasetMeta(container) {
        if (!this.datasetInfo) return;

        const columns = this.datasetInfo.columns || [];
        const totalRows = this.datasetInfo.total || 0;

        container.innerHTML = `
            <div class="dataset-meta">
                <div class="dataset-meta-item">
                    <span class="dataset-meta-label">Rows:</span>
                    <span class="dataset-meta-value">${totalRows.toLocaleString()}</span>
                </div>
                <div class="dataset-meta-item">
                    <span class="dataset-meta-label">Columns:</span>
                    <span class="dataset-meta-value">${columns.length}</span>
                </div>
                <div class="dataset-meta-item">
                    <span class="dataset-meta-label">File:</span>
                    <span class="dataset-meta-value">diabetes.csv</span>
                </div>
            </div>
            ${this.dvcInfo ? `
                <div class="dvc-badge" title="DVC Version: ${this.dvcInfo.md5}">
                    <span>📦 DVC</span>
                    <code>${this.dvcInfo.md5.substring(0, 8)}...</code>
                </div>
            ` : ''}
        `;
    }

    /**
     * Render the data table
     * @param {HTMLElement} container - Container element
     */
    renderDataTable(container) {
        if (!this.datasetInfo) return;

        // Create column definitions from dataset columns
        const columns = (this.datasetInfo.columns || []).map(col => ({
            key: col,
            label: col,
            type: this.getColumnType(col)
        }));

        // Initialize or update DataTable
        if (!this.dataTable) {
            this.dataTable = new DataTable({
                columns: columns,
                data: this.datasetInfo.data || [],
                pageSize: this.pageSize,
                currentPage: this.currentPage,
                totalRows: this.datasetInfo.total || 0,
                onPageChange: (page) => this.handlePageChange(page),
                onFilter: (value) => this.handleFilter(value)
            });
        } else {
            this.dataTable.updateData({
                columns: columns,
                data: this.datasetInfo.data || [],
                currentPage: this.currentPage,
                totalRows: this.datasetInfo.total || 0
            });
        }

        container.innerHTML = '';
        this.dataTable.render(container);
    }

    /**
     * Get column type based on column name
     * @param {string} columnName - Column name
     * @returns {string} - Column type
     */
    getColumnType(columnName) {
        // All columns in diabetes dataset are numeric
        const numericColumns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ];
        return numericColumns.includes(columnName) ? 'number' : 'string';
    }

    /**
     * Handle page change
     * @param {number} page - New page number
     */
    async handlePageChange(page) {
        this.currentPage = page;
        
        const tableContainer = document.getElementById('dataset-table-container');
        if (tableContainer) {
            tableContainer.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading page ${page}...</span>
                </div>
            `;
        }

        try {
            const datasetResponse = await dataLoader.fetchDataset(page, this.pageSize, true);
            this.datasetInfo = datasetResponse;
            
            if (this.dataTable) {
                this.dataTable.updateData({
                    data: datasetResponse.data || [],
                    currentPage: page,
                    totalRows: datasetResponse.total || 0
                });
            }
            
            if (tableContainer) {
                tableContainer.innerHTML = '';
                this.dataTable.render(tableContainer);
            }
        } catch (error) {
            console.error('Failed to load page:', error);
            if (tableContainer) {
                tableContainer.innerHTML = `
                    <div class="error-message">
                        <span>⚠️</span>
                        <span>Failed to load page ${page}: ${error.message}</span>
                    </div>
                `;
            }
        }
    }

    /**
     * Handle filter change
     * @param {string} value - Filter value
     */
    handleFilter(value) {
        console.log('Filter applied:', value);
        // Filter is handled client-side by DataTable component
    }

    /**
     * Render error state
     * @param {HTMLElement} metaContainer - Meta container element
     * @param {HTMLElement} tableContainer - Table container element
     * @param {Error} error - Error object
     */
    renderError(metaContainer, tableContainer, error) {
        const errorMessage = error instanceof DataLoaderError 
            ? error.getUserMessage() 
            : (error.message || 'Failed to load dataset');

        metaContainer.innerHTML = `
            <div class="error-message">
                <span>⚠️</span>
                <span>Failed to load dataset information</span>
            </div>
        `;

        tableContainer.innerHTML = `
            <div class="retry-section">
                <div class="error-message">
                    <span>⚠️</span>
                    <span>${errorMessage}</span>
                </div>
                <button class="btn btn-retry" id="retry-dataset-btn">
                    🔄 Try Again
                </button>
            </div>
        `;

        const retryBtn = tableContainer.querySelector('#retry-dataset-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.loadDataset(false));
        }
    }

    /**
     * Setup event listeners for panel lifecycle
     * @param {HTMLElement} panel - The panel element
     */
    setupEventListeners(panel) {
        panel.addEventListener('panel:activate', () => this.onActivate());
        panel.addEventListener('panel:deactivate', () => this.onDeactivate());
    }

    /**
     * Called when the panel becomes visible
     */
    onActivate() {
        console.log('Dataset panel activated');
        // Load dataset when panel is activated
        if (!this.datasetInfo && !this.isLoading) {
            this.loadDataset();
        }
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('Dataset panel deactivated');
    }
}

// Initialize the dataset panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const datasetPanel = new DatasetPanel();
    datasetPanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.DatasetPanel = DatasetPanel;
}
