/**
 * Navigation Component
 * Handles sidebar navigation rendering and section switching
 */
class NavigationComponent {
    constructor(config) {
        this.items = config.items || [];
        this.activeSection = config.activeSection || 'home';
        this.onNavigate = config.onNavigate || (() => {});
        this.container = null;
    }

    /**
     * Default navigation items for the ML Dashboard
     */
    static getDefaultItems() {
        return [
            { id: 'home', label: 'Home', icon: '🏠' },
            { id: 'api', label: 'API', icon: '🔌' },
            { id: 'eda', label: 'EDA', icon: '📊' },
            { id: 'mlflow', label: 'MLflow', icon: '🧪' },
            { id: 'database', label: 'Database', icon: '💾' },
            { id: 'dataset', label: 'Dataset', icon: '📋' },
            { id: 'featurestore', label: 'Feature Store', icon: '🗄️' }
        ];
    }

    /**
     * Initialize the navigation component
     * @param {string} containerId - ID of the container element
     */
    init(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Navigation container #${containerId} not found`);
            return;
        }
        this.render();
    }

    /**
     * Render the navigation items
     */
    render() {
        const navList = this.container.querySelector('#nav-list') || this.container;
        navList.innerHTML = '';

        this.items.forEach(item => {
            const navItem = this.createNavItem(item);
            navList.appendChild(navItem);
        });
    }

    /**
     * Create a navigation item element
     * @param {Object} item - Navigation item configuration
     * @returns {HTMLElement} - The navigation item element
     */
    createNavItem(item) {
        const li = document.createElement('li');
        li.className = 'nav-item';
        li.setAttribute('data-section', item.id);

        const link = document.createElement('a');
        link.className = `nav-link${item.id === this.activeSection ? ' active' : ''}`;
        link.href = '#';
        link.setAttribute('data-section', item.id);

        const icon = document.createElement('span');
        icon.className = 'nav-icon';
        icon.textContent = item.icon;

        const label = document.createElement('span');
        label.className = 'nav-label';
        label.textContent = item.label;

        link.appendChild(icon);
        link.appendChild(label);
        li.appendChild(link);

        // Add click event handler
        link.addEventListener('click', (e) => {
            e.preventDefault();
            this.handleNavClick(item.id);
        });

        return li;
    }

    /**
     * Handle navigation click events
     * @param {string} sectionId - The section ID that was clicked
     */
    handleNavClick(sectionId) {
        if (sectionId === this.activeSection) {
            return; // Already on this section
        }

        this.setActiveSection(sectionId);
        this.onNavigate(sectionId);
    }

    /**
     * Set the active section and update styling
     * @param {string} sectionId - The section ID to activate
     */
    setActiveSection(sectionId) {
        this.activeSection = sectionId;
        this.updateActiveState();
    }

    /**
     * Update the active state styling for navigation items
     */
    updateActiveState() {
        if (!this.container) return;

        const links = this.container.querySelectorAll('.nav-link');
        links.forEach(link => {
            const linkSection = link.getAttribute('data-section');
            if (linkSection === this.activeSection) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    /**
     * Get the current active section
     * @returns {string} - The active section ID
     */
    getActiveSection() {
        return this.activeSection;
    }

    /**
     * Get all navigation items
     * @returns {Array} - Array of navigation items
     */
    getItems() {
        return this.items;
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.NavigationComponent = NavigationComponent;
}
