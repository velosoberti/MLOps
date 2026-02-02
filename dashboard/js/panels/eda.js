/**
 * EDA Panel
 * Embeds the Streamlit EDA dashboard in an iframe
 */
class EDAPanel {
    constructor() {
        this.panelId = 'eda';
        this.container = null;
        this.streamlitUrl = 'http://localhost:8501';
        this.iframe = null;
        this.isLoading = true;
        this.hasError = false;
    }

    /**
     * Initialize the EDA panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('EDA panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('EDA panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the EDA panel content
     */
    render() {
        this.container.innerHTML = '';

        // Header section
        const headerSection = this.renderHeaderSection();
        this.container.appendChild(headerSection);

        // Iframe container
        const iframeSection = this.renderIframeSection();
        this.container.appendChild(iframeSection);
    }

    /**
     * Render the header section
     * @returns {HTMLElement} - Header section element
     */
    renderHeaderSection() {
        const section = document.createElement('div');
        section.className = 'card eda-header-card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📊 Exploratory Data Analysis</h3>
                <div class="eda-actions">
                    <button class="btn btn-outline" id="refresh-eda-btn" title="Refresh">
                        🔄 Refresh
                    </button>
                    <a href="${this.streamlitUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
                        Open in New Tab ↗
                    </a>
                </div>
            </div>
            <div class="card-body">
                <p class="text-muted">
                    Interactive Exploratory Data Analysis powered by Streamlit. 
                    Visualize dataset distributions, correlations, and feature relationships.
                </p>
                <div class="eda-info mt-md">
                    <div class="info-item">
                        <span class="info-label">Streamlit URL</span>
                        <code class="info-value">${this.streamlitUrl}</code>
                    </div>
                </div>
            </div>
        `;

        // Setup refresh button
        setTimeout(() => {
            const refreshBtn = section.querySelector('#refresh-eda-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.refreshIframe());
            }
        }, 0);

        return section;
    }

    /**
     * Render the iframe section
     * @returns {HTMLElement} - Iframe section element
     */
    renderIframeSection() {
        const section = document.createElement('div');
        section.className = 'card eda-iframe-card';
        section.id = 'eda-iframe-container';

        section.innerHTML = `
            <div class="iframe-wrapper">
                <div class="iframe-loading" id="iframe-loading">
                    <div class="loading">
                        <div class="loading-spinner large"></div>
                        <span class="loading-text">Loading Streamlit Dashboard...</span>
                    </div>
                </div>
                <div class="iframe-error hidden" id="iframe-error">
                    <div class="error-container">
                        <div class="error-icon">⚠️</div>
                        <div class="error-title">Unable to load Streamlit Dashboard</div>
                        <div class="error-description">
                            Please ensure Streamlit is running at <code>${this.streamlitUrl}</code>
                        </div>
                        <div class="error-instructions mt-md">
                            <p><strong>To start Streamlit:</strong></p>
                            <pre class="code-block">cd eda_streamlit && streamlit run eda.py</pre>
                        </div>
                        <div class="error-actions mt-md">
                            <button class="btn btn-primary" id="retry-iframe-btn">
                                🔄 Try Again
                            </button>
                            <a href="${this.streamlitUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-outline">
                                Open in New Tab ↗
                            </a>
                        </div>
                    </div>
                </div>
                <iframe 
                    id="streamlit-iframe"
                    src="${this.streamlitUrl}"
                    class="streamlit-iframe"
                    title="Streamlit EDA Dashboard"
                    frameborder="0"
                    allowfullscreen
                ></iframe>
            </div>
        `;

        // Setup iframe event listeners
        setTimeout(() => this.setupIframeListeners(), 0);

        return section;
    }

    /**
     * Setup iframe event listeners
     */
    setupIframeListeners() {
        this.iframe = document.getElementById('streamlit-iframe');
        const loadingEl = document.getElementById('iframe-loading');
        const errorEl = document.getElementById('iframe-error');
        const retryBtn = document.getElementById('retry-iframe-btn');

        if (this.iframe) {
            this.iframe.addEventListener('load', () => {
                this.isLoading = false;
                if (loadingEl) loadingEl.classList.add('hidden');
                // Check if iframe loaded successfully
                this.checkIframeContent();
            });

            this.iframe.addEventListener('error', () => {
                this.showError();
            });
        }

        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.refreshIframe());
        }

        // Set a timeout to show error if iframe doesn't load
        setTimeout(() => {
            if (this.isLoading) {
                this.showError();
            }
        }, 10000); // 10 second timeout
    }

    /**
     * Check if iframe content loaded successfully
     */
    checkIframeContent() {
        try {
            // Try to access iframe content - this will fail if cross-origin
            // but that's okay, it means the content loaded
            const iframeDoc = this.iframe.contentDocument || this.iframe.contentWindow.document;
            if (iframeDoc && iframeDoc.body && iframeDoc.body.innerHTML === '') {
                this.showError();
            }
        } catch (e) {
            // Cross-origin error means content loaded from different origin
            // This is expected for Streamlit, so we consider it a success
            console.log('Streamlit iframe loaded (cross-origin)');
        }
    }

    /**
     * Show error state
     */
    showError() {
        this.hasError = true;
        this.isLoading = false;
        
        const loadingEl = document.getElementById('iframe-loading');
        const errorEl = document.getElementById('iframe-error');
        
        if (loadingEl) loadingEl.classList.add('hidden');
        if (errorEl) errorEl.classList.remove('hidden');
        if (this.iframe) this.iframe.classList.add('hidden');
    }

    /**
     * Refresh the iframe
     */
    refreshIframe() {
        this.isLoading = true;
        this.hasError = false;

        const loadingEl = document.getElementById('iframe-loading');
        const errorEl = document.getElementById('iframe-error');

        if (loadingEl) loadingEl.classList.remove('hidden');
        if (errorEl) errorEl.classList.add('hidden');
        if (this.iframe) {
            this.iframe.classList.remove('hidden');
            this.iframe.src = this.streamlitUrl + '?refresh=' + Date.now();
        }

        // Set timeout for error
        setTimeout(() => {
            if (this.isLoading) {
                this.showError();
            }
        }, 10000);
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
        console.log('EDA panel activated');
        // Refresh iframe when panel is activated if it had an error
        if (this.hasError) {
            this.refreshIframe();
        }
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('EDA panel deactivated');
    }
}

// Initialize the EDA panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const edaPanel = new EDAPanel();
    edaPanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.EDAPanel = EDAPanel;
}
