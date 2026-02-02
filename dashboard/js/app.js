/**
 * Main Application Controller
 * Initializes and coordinates all dashboard components
 */
class DashboardApp {
    constructor() {
        this.navigation = null;
        this.currentSection = 'home';
        this.previousSection = null;
        this.panels = {};
        this.panelInstances = {};
        this.isTransitioning = false;
        this.panelTitles = {
            home: 'Home',
            api: 'API Documentation',
            eda: 'Exploratory Data Analysis',
            mlflow: 'MLflow Experiments',
            database: 'Database & Versioning',
            dataset: 'Dataset Viewer',
            featurestore: 'Feature Store'
        };
    }

    /**
     * Initialize the dashboard application
     */
    init() {
        this.initNavigation();
        this.initPanels();
        this.registerPanelInstances();
        this.showPanel(this.currentSection);
        this.updatePanelTitle(this.currentSection);
        console.log('ML Dashboard initialized');
    }

    /**
     * Initialize the navigation component
     */
    initNavigation() {
        const navItems = NavigationComponent.getDefaultItems();
        
        this.navigation = new NavigationComponent({
            items: navItems,
            activeSection: this.currentSection,
            onNavigate: (sectionId) => this.handleNavigation(sectionId)
        });

        this.navigation.init('navigation');
    }

    /**
     * Initialize panel references
     */
    initPanels() {
        const panelIds = ['home', 'api', 'eda', 'mlflow', 'database', 'dataset', 'featurestore'];
        
        panelIds.forEach(id => {
            const panel = document.getElementById(`panel-${id}`);
            if (panel) {
                this.panels[id] = panel;
            }
        });
    }

    /**
     * Register panel class instances for lifecycle management
     */
    registerPanelInstances() {
        // Map panel IDs to their global instances
        const panelMapping = {
            home: 'HomePanel',
            api: 'APIPanel',
            mlflow: 'MLflowPanel',
            database: 'DatabasePanel',
            dataset: 'DatasetPanel',
            featurestore: 'FeatureStorePanel'
        };

        Object.entries(panelMapping).forEach(([id, className]) => {
            // Check if the panel class exists on window
            if (window[className]) {
                this.panelInstances[id] = {
                    className,
                    initialized: true
                };
            }
        });
    }

    /**
     * Handle navigation events
     * @param {string} sectionId - The section to navigate to
     */
    handleNavigation(sectionId) {
        // Prevent navigation during transition
        if (this.isTransitioning) {
            return;
        }

        // Validate section exists
        if (!this.panels[sectionId]) {
            console.warn(`Panel for section "${sectionId}" not found, defaulting to home`);
            sectionId = 'home';
        }

        // Skip if already on this section
        if (sectionId === this.currentSection) {
            return;
        }

        this.previousSection = this.currentSection;
        this.currentSection = sectionId;
        this.showPanel(sectionId);
        this.updatePanelTitle(sectionId);
    }

    /**
     * Show a specific panel and hide others with smooth transition
     * @param {string} sectionId - The section ID to show
     */
    showPanel(sectionId) {
        this.isTransitioning = true;

        // Deactivate previous panel
        if (this.previousSection && this.panels[this.previousSection]) {
            const prevPanel = this.panels[this.previousSection];
            prevPanel.classList.add('panel-exit');
            this.triggerPanelLifecycle(prevPanel, 'deactivate');
            
            // Remove exit class after animation
            setTimeout(() => {
                prevPanel.classList.remove('panel-exit');
                prevPanel.classList.add('hidden');
            }, 150);
        }

        // Hide all other panels immediately (except previous which is animating)
        Object.entries(this.panels).forEach(([id, panel]) => {
            if (id !== sectionId && id !== this.previousSection) {
                panel.classList.add('hidden');
                panel.classList.remove('panel-enter', 'panel-exit');
            }
        });

        // Show and activate the requested panel
        const activePanel = this.panels[sectionId];
        if (activePanel) {
            // Small delay to allow exit animation
            setTimeout(() => {
                activePanel.classList.remove('hidden');
                activePanel.classList.add('panel-enter');
                this.triggerPanelLifecycle(activePanel, 'activate');
                
                // Remove enter class after animation
                setTimeout(() => {
                    activePanel.classList.remove('panel-enter');
                    this.isTransitioning = false;
                }, 250);
            }, this.previousSection ? 100 : 0);
        } else {
            this.isTransitioning = false;
        }
    }

    /**
     * Trigger panel lifecycle events
     * @param {HTMLElement} panel - The panel element
     * @param {string} event - The lifecycle event ('activate' or 'deactivate')
     */
    triggerPanelLifecycle(panel, event) {
        const panelId = panel.getAttribute('data-panel');
        const customEvent = new CustomEvent(`panel:${event}`, {
            detail: { panelId, timestamp: Date.now() }
        });
        panel.dispatchEvent(customEvent);
    }

    /**
     * Update the panel title in the header
     * @param {string} sectionId - The section ID
     */
    updatePanelTitle(sectionId) {
        const titleElement = document.getElementById('panel-title');
        if (titleElement) {
            titleElement.textContent = this.panelTitles[sectionId] || sectionId;
        }
    }

    /**
     * Get the current active section
     * @returns {string} - The current section ID
     */
    getCurrentSection() {
        return this.currentSection;
    }

    /**
     * Get the previous section
     * @returns {string|null} - The previous section ID or null
     */
    getPreviousSection() {
        return this.previousSection;
    }

    /**
     * Navigate to a specific section programmatically
     * @param {string} sectionId - The section to navigate to
     */
    navigateTo(sectionId) {
        if (this.navigation) {
            this.navigation.setActiveSection(sectionId);
        }
        this.handleNavigation(sectionId);
    }

    /**
     * Check if a panel is currently active
     * @param {string} sectionId - The section ID to check
     * @returns {boolean} - True if the panel is active
     */
    isPanelActive(sectionId) {
        return this.currentSection === sectionId;
    }

    /**
     * Get all registered panel IDs
     * @returns {string[]} - Array of panel IDs
     */
    getPanelIds() {
        return Object.keys(this.panels);
    }

    /**
     * Refresh the current panel's data
     */
    refreshCurrentPanel() {
        const activePanel = this.panels[this.currentSection];
        if (activePanel) {
            this.triggerPanelLifecycle(activePanel, 'deactivate');
            this.triggerPanelLifecycle(activePanel, 'activate');
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardApp = new DashboardApp();
    window.dashboardApp.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.DashboardApp = DashboardApp;
}
