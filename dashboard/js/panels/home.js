/**
 * Home Panel
 * Displays service cards for Flask API, Streamlit EDA, and MLflow
 */
class HomePanel {
    constructor() {
        this.panelId = 'home';
        this.container = null;
        this.services = this.getServiceConfigs();
    }

    /**
     * Get service configurations for all ML project services
     * @returns {Object[]} - Array of service configurations
     */
    getServiceConfigs() {
        return [
            {
                title: 'Flask API',
                description: 'Prediction API for the diabetes classification model. Provides endpoints for health checks, model information, and batch predictions.',
                url: 'http://localhost:5005',
                port: 5005,
                icon: '🔌',
                endpoints: ['/health', '/model/info', '/predict', '/predict/batch', '/model/reload']
            },
            {
                title: 'Streamlit EDA',
                description: 'Interactive Exploratory Data Analysis application. Visualize dataset distributions, correlations, and feature relationships.',
                url: 'http://localhost:8501',
                port: 8501,
                icon: '📊',
                endpoints: []
            },
            {
                title: 'MLflow',
                description: 'Experiment tracking and model registry. View training runs, compare metrics, and manage model versions.',
                url: 'http://127.0.0.1:5000',
                port: 5000,
                icon: '🧪',
                endpoints: []
            }
        ];
    }

    /**
     * Initialize the home panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('Home panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('Home panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the home panel content
     */
    render() {
        this.container.innerHTML = '';

        // Welcome section
        const welcomeSection = this.renderWelcomeSection();
        this.container.appendChild(welcomeSection);

        // Service cards grid
        const cardsGrid = document.createElement('div');
        cardsGrid.className = 'grid-auto';

        this.services.forEach(serviceConfig => {
            const card = new ServiceCard(serviceConfig);
            cardsGrid.appendChild(card.render());
        });

        this.container.appendChild(cardsGrid);

        // Quick info section
        const infoSection = this.renderInfoSection();
        this.container.appendChild(infoSection);
    }

    /**
     * Render the welcome section
     * @returns {HTMLElement} - Welcome section element
     */
    renderWelcomeSection() {
        const section = document.createElement('div');
        section.className = 'card welcome-section';

        section.innerHTML = `
            <div class="card-body">
                <h3 class="welcome-title">Welcome to the ML Project Dashboard</h3>
                <p class="welcome-text text-muted">
                    This dashboard provides a unified interface to monitor and access all components of your 
                    diabetes prediction ML project. Click on any service card below to open it in a new tab, 
                    or use the navigation sidebar to explore detailed information about each component.
                </p>
            </div>
        `;

        return section;
    }

    /**
     * Render the quick info section
     * @returns {HTMLElement} - Info section element
     */
    renderInfoSection() {
        const section = document.createElement('div');
        section.className = 'card info-section mt-lg';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📌 Quick Info</h3>
            </div>
            <div class="card-body">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Model</span>
                        <span class="info-value">diabete_model</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Experiment ID</span>
                        <span class="info-value">467326610704772702</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Dataset</span>
                        <span class="info-value">diabetes.csv</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Feature Store</span>
                        <span class="info-value">Feast (SQLite)</span>
                    </div>
                </div>
            </div>
        `;

        return section;
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
        console.log('Home panel activated');
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('Home panel deactivated');
    }
}

// Initialize the home panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const homePanel = new HomePanel();
    homePanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.HomePanel = HomePanel;
}
