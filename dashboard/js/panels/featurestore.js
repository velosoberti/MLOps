/**
 * Feature Store Info Panel
 * Displays Feast feature store configuration, entities, and feature views
 */
class FeatureStorePanel {
    constructor() {
        this.panelId = 'featurestore';
        this.container = null;
        this.config = null;
        this.views = null;
        this.isLoading = false;
        this.error = null;
    }

    /**
     * Initialize the feature store panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('Feature Store panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('Feature Store panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the feature store panel content
     */
    render() {
        this.container.innerHTML = '';

        // Overview section
        const overviewSection = this.renderOverviewSection();
        this.container.appendChild(overviewSection);

        // Configuration section
        const configSection = this.renderConfigSection();
        this.container.appendChild(configSection);

        // Entities section
        const entitiesSection = this.renderEntitiesSection();
        this.container.appendChild(entitiesSection);

        // Feature Views section
        const featureViewsSection = this.renderFeatureViewsSection();
        this.container.appendChild(featureViewsSection);

        // Feature Data Viewer section (integrated)
        const featureDataSection = this.renderFeatureDataSection();
        this.container.appendChild(featureDataSection);
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
                <h3 class="card-title">🏪 Feature Store</h3>
                <button class="btn btn-outline" id="refresh-featurestore-btn">
                    🔄 Refresh
                </button>
            </div>
            <div class="card-body">
                <p class="text-muted">
                    This project uses Feast (Feature Store) to manage and serve ML features.
                    Feast provides a centralized repository for storing and serving features for both training and inference.
                </p>
            </div>
        `;

        // Add refresh button handler
        setTimeout(() => {
            const refreshBtn = section.querySelector('#refresh-featurestore-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.loadFeatureStoreData(false));
            }
        }, 0);

        return section;
    }

    /**
     * Render the configuration section
     * @returns {HTMLElement} - Configuration section element
     */
    renderConfigSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'featurestore-config-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">⚙️ Configuration</h3>
            </div>
            <div class="card-body" id="featurestore-config-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading configuration...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the entities section
     * @returns {HTMLElement} - Entities section element
     */
    renderEntitiesSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'featurestore-entities-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🔑 Entities</h3>
            </div>
            <div class="card-body" id="featurestore-entities-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading entities...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the feature views section
     * @returns {HTMLElement} - Feature views section element
     */
    renderFeatureViewsSection() {
        const section = document.createElement('div');
        section.className = 'card';
        section.id = 'featurestore-views-section';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📊 Feature Views</h3>
            </div>
            <div class="card-body" id="featurestore-views-content">
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <span>Loading feature views...</span>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the feature data section (integrated data viewer)
     * @returns {HTMLElement} - Feature data section element
     */
    renderFeatureDataSection() {
        const section = document.createElement('div');
        section.id = 'featurestore-data-section';

        // Create a container for the FeatureDataPanel to render into
        section.innerHTML = `
            <div id="feature-data-container"></div>
        `;

        // Initialize the FeatureDataPanel in this container after DOM is ready
        setTimeout(() => {
            if (window.featureDataPanel) {
                window.featureDataPanel.init('feature-data-container');
            }
        }, 0);

        return section;
    }

    /**
     * Load feature store data from the API
     * @param {boolean} useCache - Whether to use cached data
     */
    async loadFeatureStoreData(useCache = true) {
        const configContainer = document.getElementById('featurestore-config-content');
        const entitiesContainer = document.getElementById('featurestore-entities-content');
        const viewsContainer = document.getElementById('featurestore-views-content');

        if (!configContainer || !entitiesContainer || !viewsContainer) return;

        this.isLoading = true;
        this.error = null;

        // Show loading states using helper function if available
        const loadingHtml = window.renderLoadingState 
            ? window.renderLoadingState('Loading...')
            : `<div class="loading"><div class="loading-spinner"></div><span>Loading...</span></div>`;
        
        configContainer.innerHTML = loadingHtml;
        entitiesContainer.innerHTML = loadingHtml;
        viewsContainer.innerHTML = loadingHtml;

        try {
            // Load config and views in parallel
            const [configResponse, viewsResponse] = await Promise.all([
                dataLoader.fetchFeatureStoreConfig(useCache),
                dataLoader.fetchFeatureStoreViews(useCache)
            ]);

            this.config = configResponse;
            this.views = viewsResponse;

            this.renderConfigContent(configContainer);
            this.renderEntitiesContent(entitiesContainer);
            this.renderFeatureViewsContent(viewsContainer);
        } catch (error) {
            this.error = error;
            this.renderError(configContainer, error);
            this.renderError(entitiesContainer, error);
            this.renderError(viewsContainer, error);
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Render configuration content
     * @param {HTMLElement} container - Container element
     */
    renderConfigContent(container) {
        if (!this.config) return;

        const onlineStore = this.config.onlineStore || {};

        container.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Project Name</span>
                    <span class="info-value">${this.config.project || 'N/A'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Provider</span>
                    <span class="info-value">${this.config.provider || 'N/A'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Registry</span>
                    <span class="info-value">${this.config.registry || 'N/A'}</span>
                </div>
            </div>
            <div class="mt-md">
                <h4 class="subsection-title">Online Store Configuration</h4>
                <div class="online-store-config">
                    <div class="config-item">
                        <span class="config-label">Type:</span>
                        <span class="config-value">${onlineStore.type || 'N/A'}</span>
                    </div>
                    <div class="config-item">
                        <span class="config-label">Path:</span>
                        <span class="config-value">${onlineStore.path || 'N/A'}</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render entities content
     * @param {HTMLElement} container - Container element
     */
    renderEntitiesContent(container) {
        if (!this.views || !this.views.entities) {
            container.innerHTML = '<p class="text-muted">No entities found</p>';
            return;
        }

        const entities = this.views.entities;

        if (entities.length === 0) {
            container.innerHTML = '<p class="text-muted">No entities defined</p>';
            return;
        }

        const entitiesHtml = entities.map(entity => `
            <div class="entity-card">
                <div class="entity-header">
                    <span class="entity-name">${entity.name}</span>
                    <span class="badge badge-primary">${entity.valueType}</span>
                </div>
                ${entity.description ? `<p class="entity-description">${entity.description}</p>` : ''}
            </div>
        `).join('');

        container.innerHTML = `
            <div class="entities-list">
                ${entitiesHtml}
            </div>
        `;
    }

    /**
     * Render feature views content
     * @param {HTMLElement} container - Container element
     */
    renderFeatureViewsContent(container) {
        if (!this.views || !this.views.featureViews) {
            container.innerHTML = '<p class="text-muted">No feature views found</p>';
            return;
        }

        const featureViews = this.views.featureViews;

        if (featureViews.length === 0) {
            container.innerHTML = '<p class="text-muted">No feature views defined</p>';
            return;
        }

        const viewsHtml = featureViews.map(view => {
            const schemaHtml = (view.schema || []).map(field => `
                <div class="schema-field">
                    <span class="field-name">${field.name}</span>
                    <span class="field-type">${field.dtype}</span>
                </div>
            `).join('');

            const ttlDisplay = this.formatTTL(view.ttl);

            return `
                <div class="feature-view-card">
                    <div class="feature-view-header">
                        <h4 class="feature-view-name">${view.name}</h4>
                        <div class="feature-view-badges">
                            ${view.online ? '<span class="badge badge-success">Online</span>' : '<span class="badge">Offline</span>'}
                        </div>
                    </div>
                    <div class="feature-view-meta">
                        <div class="meta-item">
                            <span class="meta-label">TTL:</span>
                            <span class="meta-value">${ttlDisplay}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Entities:</span>
                            <span class="meta-value">${(view.entities || []).join(', ') || 'N/A'}</span>
                        </div>
                    </div>
                    <div class="feature-view-schema">
                        <h5 class="schema-title">Schema Fields (${(view.schema || []).length})</h5>
                        <div class="schema-fields">
                            ${schemaHtml || '<p class="text-muted">No schema fields</p>'}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="feature-views-list">
                ${viewsHtml}
            </div>
        `;
    }

    /**
     * Format TTL value to human-readable format
     * @param {number} ttl - TTL in seconds
     * @returns {string} - Formatted TTL string
     */
    formatTTL(ttl) {
        if (!ttl) return 'N/A';
        
        const days = Math.floor(ttl / 86400);
        const hours = Math.floor((ttl % 86400) / 3600);
        
        if (days > 0) {
            return `${days} day${days > 1 ? 's' : ''}${hours > 0 ? ` ${hours}h` : ''} (${ttl.toLocaleString()}s)`;
        } else if (hours > 0) {
            return `${hours} hour${hours > 1 ? 's' : ''} (${ttl.toLocaleString()}s)`;
        }
        return `${ttl.toLocaleString()} seconds`;
    }

    /**
     * Render error state
     * @param {HTMLElement} container - Container element
     * @param {Error} error - Error object
     */
    renderError(container, error) {
        const errorMessage = error instanceof DataLoaderError 
            ? error.getUserMessage() 
            : (error.message || 'Failed to load feature store data');

        container.innerHTML = `
            <div class="retry-section">
                <div class="error-message">
                    <span>⚠️</span>
                    <span>${errorMessage}</span>
                </div>
                <button class="btn btn-retry retry-featurestore-btn">
                    🔄 Try Again
                </button>
            </div>
        `;

        const retryBtn = container.querySelector('.retry-featurestore-btn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.loadFeatureStoreData(false));
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
        console.log('Feature Store panel activated');
        // Load feature store data when panel is activated
        if (!this.config && !this.isLoading) {
            this.loadFeatureStoreData();
        }
        // Also trigger feature data loading
        if (window.featureDataPanel && !window.featureDataPanel.featureData && !window.featureDataPanel.isLoading) {
            window.featureDataPanel.loadFeatureData();
        }
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('Feature Store panel deactivated');
    }
}

// Initialize the feature store panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const featureStorePanel = new FeatureStorePanel();
    featureStorePanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.FeatureStorePanel = FeatureStorePanel;
}
