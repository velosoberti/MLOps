/**
 * Database Panel
 * Displays DVC version info and dataset metadata
 */
class DatabasePanel {
    constructor() {
        this.panelId = 'database';
        this.container = null;
        this.dvcInfo = null;
        this.isLoading = false;
        this.error = null;
    }

    /**
     * Initialize the database panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('Database panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('Database panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the database panel content
     */
    render() {
        this.container.innerHTML = '';

        // Overview section
        const overviewSection = this.renderOverviewSection();
        this.container.appendChild(overviewSection);

        // DVC info section
        const dvcSection = this.renderDVCSection();
        this.container.appendChild(dvcSection);

        // Dataset metadata section
        const metadataSection = this.renderMetadataSection();
        this.container.appendChild(metadataSection);
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
                <h3 class="card-title">💾 Database & Versioning</h3>
            </div>
            <div class="card-body">
                <p class="text-muted">
                    This project uses DVC (Data Version Control) to track and version the diabetes dataset.
                    DVC enables reproducible ML experiments by versioning data alongside code.
                </p>
            </div>
        `;

        return section;
    }

    /**
     * Render the DVC information section
     * @returns {HTMLElement} - DVC section element
     */
    renderDVCSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'dvc-info-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📦 DVC Version Information</h3>
                <button class="btn btn-outline" id="refresh-dvc-btn">
                    🔄 Refresh
                </button>
            </div>
            <div class="card-body" id="dvc-info-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading DVC information...</span>
                </div>
            </div>
        `;

        // Add refresh button handler
        setTimeout(() => {
            const refreshBtn = section.querySelector('#refresh-dvc-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.loadDVCInfo(false));
            }
        }, 0);

        return section;
    }

    /**
     * Render the dataset metadata section
     * @returns {HTMLElement} - Metadata section element
     */
    renderMetadataSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📋 Dataset Metadata</h3>
            </div>
            <div class="card-body">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Dataset Name</span>
                        <span class="info-value">diabetes.csv</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Location</span>
                        <span class="info-value">data/diabetes.csv</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Format</span>
                        <span class="info-value">CSV (Comma-Separated Values)</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Version Control</span>
                        <span class="info-value">DVC (Data Version Control)</span>
                    </div>
                </div>
                <div class="mt-md">
                    <p class="text-muted">
                        The diabetes dataset contains medical predictor variables and one target variable (Outcome).
                        Features include Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                        DiabetesPedigreeFunction, and Age.
                    </p>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Load DVC information from the API
     * @param {boolean} useCache - Whether to use cached data
     */
    async loadDVCInfo(useCache = true) {
        const contentContainer = document.getElementById('dvc-info-content');
        if (!contentContainer) return;

        this.isLoading = true;
        this.error = null;

        // Show loading state
        contentContainer.innerHTML = window.renderLoadingState 
            ? window.renderLoadingState('Loading DVC information...')
            : `<div class="loading"><div class="loading-spinner"></div><span>Loading DVC information...</span></div>`;

        try {
            this.dvcInfo = await dataLoader.fetchDVCInfo(useCache);
            this.renderDVCInfo(contentContainer);
        } catch (error) {
            this.error = error;
            this.renderDVCError(contentContainer, error);
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Render DVC information
     * @param {HTMLElement} container - Container element
     */
    renderDVCInfo(container) {
        if (!this.dvcInfo) return;

        const formattedSize = this.formatFileSize(this.dvcInfo.size);

        container.innerHTML = `
            <div class="dvc-info-grid">
                <div class="dvc-info-item">
                    <span class="dvc-info-label">MD5 Hash</span>
                    <span class="dvc-info-value">${this.dvcInfo.md5}</span>
                </div>
                <div class="dvc-info-item">
                    <span class="dvc-info-label">File Size</span>
                    <span class="dvc-info-value">${formattedSize} (${this.dvcInfo.size.toLocaleString()} bytes)</span>
                </div>
                <div class="dvc-info-item">
                    <span class="dvc-info-label">File Path</span>
                    <span class="dvc-info-value">${this.dvcInfo.path}</span>
                </div>
                <div class="dvc-info-item">
                    <span class="dvc-info-label">Hash Algorithm</span>
                    <span class="dvc-info-value">${this.dvcInfo.hash || 'md5'}</span>
                </div>
            </div>
            <div class="mt-md">
                <p class="text-muted">
                    The MD5 hash uniquely identifies this version of the dataset. Any changes to the data
                    will result in a different hash, enabling precise version tracking.
                </p>
            </div>
        `;
    }

    /**
     * Render DVC error state
     * @param {HTMLElement} container - Container element
     * @param {Error} error - Error object
     */
    renderDVCError(container, error) {
        const errorMessage = error instanceof DataLoaderError 
            ? error.getUserMessage() 
            : (error.message || 'Failed to load DVC information');
        
        container.innerHTML = `
            <div class="retry-section">
                <div class="error-message">
                    <span>⚠️</span>
                    <span>${errorMessage}</span>
                </div>
                <button class="btn btn-retry" id="retry-dvc-btn">
                    🔄 Try Again
                </button>
            </div>
        `;

        const retryBtn = container.querySelector('#retry-dvc-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.loadDVCInfo(false));
        }
    }

    /**
     * Format file size to human-readable format
     * @param {number} bytes - File size in bytes
     * @returns {string} - Formatted file size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
        console.log('Database panel activated');
        // Load DVC info when panel is activated
        if (!this.dvcInfo && !this.isLoading) {
            this.loadDVCInfo();
        }
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('Database panel deactivated');
    }
}

// Initialize the database panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const databasePanel = new DatabasePanel();
    databasePanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.DatabasePanel = DatabasePanel;
}
