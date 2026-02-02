/**
 * API Documentation Panel
 * Displays Flask API endpoints, methods, descriptions, examples, and interactive prediction form
 */
class APIPanel {
    constructor() {
        this.panelId = 'api';
        this.container = null;
        this.apiBaseUrl = 'http://localhost:5005';
        this.endpoints = this.getEndpointConfigs();
        this.inputFeatures = this.getInputFeatures();
        this.examplePatients = this.getExamplePatients();
        this.isLoading = false;
    }

    /**
     * Get example patient data for quick testing
     * @returns {Object} - Example patients
     */
    getExamplePatients() {
        return {
            alto_risco: {
                name: "Alto Risco",
                data: {
                    Glucose: 180,
                    BMI: 38.5,
                    DiabetesPedigreeFunction: 0.85,
                    Insulin: 150,
                    SkinThickness: 40
                }
            },
            medio_risco: {
                name: "Médio Risco",
                data: {
                    Glucose: 120,
                    BMI: 30.2,
                    DiabetesPedigreeFunction: 0.45,
                    Insulin: 80,
                    SkinThickness: 28
                }
            },
            baixo_risco: {
                name: "Baixo Risco",
                data: {
                    Glucose: 85,
                    BMI: 24.5,
                    DiabetesPedigreeFunction: 0.25,
                    Insulin: 40,
                    SkinThickness: 20
                }
            },
            valores_normais: {
                name: "Valores Normais",
                data: {
                    Glucose: 95,
                    BMI: 26.0,
                    DiabetesPedigreeFunction: 0.30,
                    Insulin: 50,
                    SkinThickness: 25
                }
            }
        };
    }

    /**
     * Get API endpoint configurations
     * @returns {Object[]} - Array of endpoint configurations
     */
    getEndpointConfigs() {
        return [
            {
                path: '/health',
                method: 'GET',
                description: 'Health check endpoint to verify the API is running and responsive.',
                hasExample: false
            },
            {
                path: '/model/info',
                method: 'GET',
                description: 'Returns information about the currently loaded model including name, version, and metadata.',
                hasExample: false
            },
            {
                path: '/predict',
                method: 'POST',
                description: 'Make a single prediction using the diabetes classification model. Accepts patient features and returns a prediction.',
                hasExample: true,
                requestExample: {
                    Glucose: 148,
                    BMI: 33.6,
                    DiabetesPedigreeFunction: 0.627,
                    Insulin: 0,
                    SkinThickness: 35
                },
                responseExample: {
                    prediction: "diabetes",
                    score: 0.73,
                    confidence: 0.73,
                    model_version: "1.0"
                }
            },
            {
                path: '/predict/batch',
                method: 'POST',
                description: 'Make batch predictions for multiple patients. Accepts an array of patient feature sets and returns predictions for each.',
                hasExample: false
            },
            {
                path: '/model/reload',
                method: 'POST',
                description: 'Reload the model from disk. Useful when the model has been updated and needs to be refreshed without restarting the server.',
                hasExample: false
            }
        ];
    }

    /**
     * Get expected input features for the prediction endpoint
     * @returns {Object[]} - Array of feature definitions
     */
    getInputFeatures() {
        return [
            { name: 'Glucose', type: 'number', description: 'Plasma glucose concentration (mg/dL)', min: 0, max: 300, step: 1 },
            { name: 'BMI', type: 'number', description: 'Body mass index (weight in kg/(height in m)²)', min: 10, max: 70, step: 0.1 },
            { name: 'DiabetesPedigreeFunction', type: 'number', description: 'Diabetes pedigree function score', min: 0, max: 3, step: 0.01 },
            { name: 'Insulin', type: 'number', description: '2-Hour serum insulin (mu U/ml)', min: 0, max: 900, step: 1 },
            { name: 'SkinThickness', type: 'number', description: 'Triceps skin fold thickness (mm)', min: 0, max: 100, step: 1 }
        ];
    }

    /**
     * Initialize the API panel
     */
    init() {
        const panel = document.getElementById(`panel-${this.panelId}`);
        if (!panel) {
            console.error('API panel container not found');
            return;
        }

        this.container = panel.querySelector('.panel-content');
        if (!this.container) {
            console.error('API panel content container not found');
            return;
        }

        this.render();
        this.setupEventListeners(panel);
    }

    /**
     * Render the API panel content
     */
    render() {
        this.container.innerHTML = '';

        // API Overview section
        const overviewSection = this.renderOverviewSection();
        this.container.appendChild(overviewSection);

        // Interactive Prediction section
        const predictionSection = this.renderPredictionSection();
        this.container.appendChild(predictionSection);

        // Endpoints section
        const endpointsSection = this.renderEndpointsSection();
        this.container.appendChild(endpointsSection);

        // Input Features section
        const featuresSection = this.renderFeaturesSection();
        this.container.appendChild(featuresSection);
    }

    /**
     * Render the API overview section
     * @returns {HTMLElement} - Overview section element
     */
    renderOverviewSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🔌 Flask Prediction API</h3>
                <a href="${this.apiBaseUrl}" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
                    Open API ↗
                </a>
            </div>
            <div class="card-body">
                <p class="text-muted">
                    The Flask API provides RESTful endpoints for the diabetes classification model. 
                    Use these endpoints to check API health, get model information, and make predictions.
                </p>
                <div class="api-base-url mt-md">
                    <span class="info-label">Base URL</span>
                    <code class="info-value">${this.apiBaseUrl}</code>
                </div>
            </div>
        `;

        return section;
    }

    /**
     * Render the interactive prediction section
     * @returns {HTMLElement} - Prediction section element
     */
    renderPredictionSection() {
        const section = document.createElement('div');
        section.className = 'card prediction-section';

        const exampleOptions = Object.entries(this.examplePatients)
            .map(([key, value]) => `<option value="${key}">${value.name}</option>`)
            .join('');

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">🧪 Try the API</h3>
                <div class="example-selector">
                    <label for="example-select">Load Example:</label>
                    <select id="example-select" class="input">
                        <option value="">-- Select --</option>
                        ${exampleOptions}
                    </select>
                </div>
            </div>
            <div class="card-body">
                <form id="prediction-form" class="prediction-form">
                    <div class="form-grid">
                        ${this.inputFeatures.map(feature => `
                            <div class="form-group">
                                <label for="input-${feature.name}">${feature.name}</label>
                                <input 
                                    type="number" 
                                    id="input-${feature.name}" 
                                    name="${feature.name}"
                                    class="input"
                                    min="${feature.min}"
                                    max="${feature.max}"
                                    step="${feature.step}"
                                    placeholder="${feature.description}"
                                    required
                                />
                                <span class="form-hint">${feature.description}</span>
                            </div>
                        `).join('')}
                    </div>
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary" id="predict-btn">
                            🔮 Make Prediction
                        </button>
                        <button type="button" class="btn btn-outline" id="clear-form-btn">
                            Clear
                        </button>
                    </div>
                </form>
                <div id="prediction-result" class="prediction-result hidden">
                    <!-- Result will be rendered here -->
                </div>
            </div>
        `;

        // Setup form event listeners after rendering
        setTimeout(() => this.setupFormListeners(), 0);

        return section;
    }

    /**
     * Setup form event listeners
     */
    setupFormListeners() {
        const form = document.getElementById('prediction-form');
        const exampleSelect = document.getElementById('example-select');
        const clearBtn = document.getElementById('clear-form-btn');

        if (form) {
            form.addEventListener('submit', (e) => this.handlePrediction(e));
        }

        if (exampleSelect) {
            exampleSelect.addEventListener('change', (e) => this.loadExample(e.target.value));
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearForm());
        }
    }

    /**
     * Load example patient data into the form
     * @param {string} exampleKey - Key of the example to load
     */
    loadExample(exampleKey) {
        if (!exampleKey || !this.examplePatients[exampleKey]) return;

        const data = this.examplePatients[exampleKey].data;
        
        Object.entries(data).forEach(([key, value]) => {
            const input = document.getElementById(`input-${key}`);
            if (input) {
                input.value = value;
            }
        });

        // Clear previous result
        const resultContainer = document.getElementById('prediction-result');
        if (resultContainer) {
            resultContainer.classList.add('hidden');
        }
    }

    /**
     * Clear the prediction form
     */
    clearForm() {
        const form = document.getElementById('prediction-form');
        if (form) {
            form.reset();
        }

        const exampleSelect = document.getElementById('example-select');
        if (exampleSelect) {
            exampleSelect.value = '';
        }

        const resultContainer = document.getElementById('prediction-result');
        if (resultContainer) {
            resultContainer.classList.add('hidden');
        }
    }

    /**
     * Handle prediction form submission
     * @param {Event} e - Form submit event
     */
    async handlePrediction(e) {
        e.preventDefault();

        if (this.isLoading) return;

        const form = e.target;
        const formData = new FormData(form);
        const data = {};

        this.inputFeatures.forEach(feature => {
            const value = formData.get(feature.name);
            data[feature.name] = parseFloat(value);
        });

        const predictBtn = document.getElementById('predict-btn');
        const resultContainer = document.getElementById('prediction-result');

        this.isLoading = true;
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="loading-spinner small"></span> Predicting...';

        try {
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            this.renderPredictionResult(resultContainer, response.ok, data, result);
        } catch (error) {
            this.renderPredictionError(resultContainer, error);
        } finally {
            this.isLoading = false;
            predictBtn.disabled = false;
            predictBtn.innerHTML = '🔮 Make Prediction';
        }
    }

    /**
     * Render prediction result
     * @param {HTMLElement} container - Result container
     * @param {boolean} success - Whether the request was successful
     * @param {Object} inputData - Input data sent
     * @param {Object} result - API response
     */
    renderPredictionResult(container, success, inputData, result) {
        container.classList.remove('hidden');

        if (success) {
            const isDiabetes = result.prediction === 'diabetes';
            const confidenceClass = result.confidence > 0.8 ? 'high' : result.confidence > 0.6 ? 'medium' : 'low';
            const confidenceLabel = result.confidence > 0.8 ? 'High' : result.confidence > 0.6 ? 'Medium' : 'Low';

            container.innerHTML = `
                <div class="result-card ${isDiabetes ? 'result-positive' : 'result-negative'}">
                    <div class="result-header">
                        <span class="result-icon">${isDiabetes ? '⚠️' : '✅'}</span>
                        <span class="result-title">Prediction: ${result.prediction.toUpperCase()}</span>
                    </div>
                    <div class="result-body">
                        <div class="result-metrics">
                            <div class="metric">
                                <span class="metric-label">Score</span>
                                <span class="metric-value">${(result.score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Confidence</span>
                                <span class="metric-value confidence-${confidenceClass}">${(result.confidence * 100).toFixed(1)}% (${confidenceLabel})</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Model Version</span>
                                <span class="metric-value">${result.model_version || 'N/A'}</span>
                            </div>
                        </div>
                        <div class="result-interpretation">
                            <h4>📋 Interpretation</h4>
                            ${isDiabetes 
                                ? `<p>High probability of diabetes (${(result.score * 100).toFixed(1)}%). ${
                                    result.confidence > 0.8 ? 'High confidence - Recommended to consult a doctor.' :
                                    result.confidence > 0.6 ? 'Medium confidence - Monitoring recommended.' :
                                    'Low confidence - Result uncertain, consider additional tests.'
                                }</p>`
                                : `<p>Low probability of diabetes (${(result.score * 100).toFixed(1)}%). ${
                                    result.confidence > 0.8 ? 'High confidence - Indicators are normal.' :
                                    result.confidence > 0.6 ? 'Medium confidence - Maintain healthy habits.' :
                                    'Low confidence - Preventive monitoring recommended.'
                                }</p>`
                            }
                        </div>
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="result-card result-error">
                    <div class="result-header">
                        <span class="result-icon">❌</span>
                        <span class="result-title">Prediction Failed</span>
                    </div>
                    <div class="result-body">
                        <p>${result.error || result.message || 'Unknown error occurred'}</p>
                        ${result.details ? `<p class="error-details">${result.details}</p>` : ''}
                    </div>
                </div>
            `;
        }
    }

    /**
     * Render prediction error
     * @param {HTMLElement} container - Result container
     * @param {Error} error - Error object
     */
    renderPredictionError(container, error) {
        container.classList.remove('hidden');
        container.innerHTML = `
            <div class="result-card result-error">
                <div class="result-header">
                    <span class="result-icon">❌</span>
                    <span class="result-title">Connection Error</span>
                </div>
                <div class="result-body">
                    <p>Unable to connect to the API. Please ensure the Flask API is running at ${this.apiBaseUrl}</p>
                    <p class="error-details">${error.message}</p>
                    <p class="mt-md">
                        <strong>To start the API:</strong><br>
                        <code>cd flask && python api.py</code>
                    </p>
                </div>
            </div>
        `;
    }

    /**
     * Render the endpoints section
     * @returns {HTMLElement} - Endpoints section element
     */
    renderEndpointsSection() {
        const section = document.createElement('div');
        section.className = 'card';

        const header = document.createElement('div');
        header.className = 'card-header';
        header.innerHTML = `<h3 class="card-title">📋 API Endpoints</h3>`;
        section.appendChild(header);

        const body = document.createElement('div');
        body.className = 'card-body';

        const endpointsList = document.createElement('div');
        endpointsList.className = 'endpoints-documentation';

        this.endpoints.forEach(endpoint => {
            const endpointCard = this.renderEndpointCard(endpoint);
            endpointsList.appendChild(endpointCard);
        });

        body.appendChild(endpointsList);
        section.appendChild(body);

        return section;
    }

    /**
     * Render a single endpoint card
     * @param {Object} endpoint - Endpoint configuration
     * @returns {HTMLElement} - Endpoint card element
     */
    renderEndpointCard(endpoint) {
        const card = document.createElement('div');
        card.className = 'endpoint-card';

        const methodClass = endpoint.method === 'GET' ? 'badge-get' : 'badge-post';

        let cardContent = `
            <div class="endpoint-header">
                <span class="badge ${methodClass}">${endpoint.method}</span>
                <code class="endpoint-path">${endpoint.path}</code>
            </div>
            <p class="endpoint-description">${endpoint.description}</p>
        `;

        if (endpoint.hasExample) {
            cardContent += `
                <div class="endpoint-examples mt-md">
                    <div class="example-section">
                        <h4 class="example-title">Request Example</h4>
                        <pre class="code-block">${JSON.stringify(endpoint.requestExample, null, 2)}</pre>
                    </div>
                    <div class="example-section mt-md">
                        <h4 class="example-title">Response Example</h4>
                        <pre class="code-block">${JSON.stringify(endpoint.responseExample, null, 2)}</pre>
                    </div>
                </div>
            `;
        }

        card.innerHTML = cardContent;
        return card;
    }

    /**
     * Render the input features section
     * @returns {HTMLElement} - Features section element
     */
    renderFeaturesSection() {
        const section = document.createElement('div');
        section.className = 'card';

        section.innerHTML = `
            <div class="card-header">
                <h3 class="card-title">📊 Expected Input Features</h3>
            </div>
            <div class="card-body">
                <p class="text-muted mb-md">
                    The /predict and /predict/batch endpoints expect the following features in the request body:
                </p>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Feature Name</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${this.inputFeatures.map(feature => `
                                <tr>
                                    <td><code>${feature.name}</code></td>
                                    <td>${feature.type}</td>
                                    <td>${feature.description}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
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
        console.log('API panel activated');
    }

    /**
     * Called when the panel is hidden
     */
    onDeactivate() {
        console.log('API panel deactivated');
    }
}

// Initialize the API panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const apiPanel = new APIPanel();
    apiPanel.init();
});

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.APIPanel = APIPanel;
}
