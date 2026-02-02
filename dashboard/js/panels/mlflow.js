/**
 * MLflow Panel
 * Displays MLflow experiment information, model details, and artifact paths
 */
class MLflowPanel {
    constructor() {
        this.panelId = 'mlflow';
        this.container = null;
        this.mlflowConfig = this.getMLflowConfig();
    }

    /**
     * Get MLflow configuration
     * @returns {Object} - MLflow configuration object
     */
    getMLflowConfig() {
        return {
            trackingUri: 'http://127.0.0.1:5000',
            experimentId: '467326610704772702',
            modelName: 'diabete_model',
            artifactPaths: {
                mlartifacts: 'mlflow/mlartifacts/',
                mlruns: 'mlflow/mlruns/'
            }
        };
    }

    /**
     * Initialize the MLflow panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('MLflow panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('MLflow panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the MLflow panel content
     */
    render() {
        this.container.innerHTML = '';

        // Overview section
        const overviewSection = this.renderOverviewSection();
        this.container.appendChild(overviewSection);

        // Model information section
        const modelSection = this.renderModelSection();
        this.container.appendChild(modelSection);

        // Artifacts section
        const artifactsSection = this.renderArtifactsSection();
        this.container.appendChild(artifactsSection);
    }

    /**
     * Render the MLflow overview section
     * @returns {HTMLElement} - Overview section element
     */
    renderOverviewSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🧪 MLflow Experiment Tracking</h3>
                <a href="${this.mlflowConfig.trackingUri}" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
                    Open MLflow UI ↗
                </a>
            </div>
            <div class="card-body">
                <p class="text-muted mb-md">
                    MLflow is used for experiment tracking, model versioning, and artifact management in this project.
                    View training runs, compare metrics, and manage model versions through the MLflow UI.
                </p>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Tracking URI</span>
                        <code class="info-value">${this.mlflowConfig.trackingUri}</code>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Experiment ID</span>
                        <code class="info-value">${this.mlflowConfig.experimentId}</code>
                    </div>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the model information section
     * @returns {HTMLElement} - Model section element
     */
    renderModelSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📦 Registered Model</h3>
            </div>
            <div class="card-body">
                <div class="model-info">
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Model Name</span>
                            <span class="info-value model-name">${this.mlflowConfig.modelName}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Status</span>
                            <span class="badge badge-success">Registered</span>
                        </div>
                    </div>
                </div>
                <div class="model-details mt-md">
                    <p class="text-muted">
                        The diabetes classification model is registered in MLflow's model registry. 
                        Use the MLflow UI to view model versions, stage transitions, and deployment history.
                    </p>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the artifacts section
     * @returns {HTMLElement} - Artifacts section element
     */
    renderArtifactsSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📁 Artifact Storage</h3>
            </div>
            <div class="card-body">
                <p class="text-muted mb-md">
                    MLflow stores experiment artifacts including model files, metrics, and parameters in the following directories:
                </p>
                <div class="artifacts-list">
                    <div class="artifact-item">
                        <div class="artifact-icon">📂</div>
                        <div class="artifact-details">
                            <span class="artifact-path">${this.mlflowConfig.artifactPaths.mlartifacts}</span>
                            <span class="artifact-description text-muted">Model artifacts and serialized models</span>
                        </div>
                    </div>
                    <div class="artifact-item">
                        <div class="artifact-icon">📂</div>
                        <div class="artifact-details">
                            <span class="artifact-path">${this.mlflowConfig.artifactPaths.mlruns}</span>
                            <span class="artifact-description text-muted">Experiment runs, metrics, and parameters</span>
                        </div>
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
        console.log('MLflow panel activated');
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('MLflow panel deactivated');
    }
}

// Initialize the MLflow panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const mlflowPanel = new MLflowPanel();
    mlflowPanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.MLflowPanel = MLflowPanel;
}
