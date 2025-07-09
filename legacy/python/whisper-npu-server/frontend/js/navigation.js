/**
 * Navigation Menu Controller
 * Handles hamburger menu functionality and navigation interactions
 */

class NavigationController {
    constructor() {
        this.isMenuOpen = false;
        this.init();
    }

    init() {
        this.hamburgerMenu = document.getElementById('hamburgerMenu');
        this.navMenu = document.getElementById('navMenu');
        
        if (this.hamburgerMenu && this.navMenu) {
            this.setupEventListeners();
            this.createOverlay();
        }
    }

    setupEventListeners() {
        // Hamburger menu click
        this.hamburgerMenu.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleMenu();
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (this.isMenuOpen && !this.navMenu.contains(e.target)) {
                this.closeMenu();
            }
        });

        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isMenuOpen) {
                this.closeMenu();
            }
        });

        // Close menu when clicking nav items
        const navItems = this.navMenu.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                this.closeMenu();
            });
        });
    }

    createOverlay() {
        // Create overlay element for mobile
        this.overlay = document.createElement('div');
        this.overlay.className = 'nav-overlay';
        this.overlay.addEventListener('click', () => {
            this.closeMenu();
        });
        document.body.appendChild(this.overlay);
    }

    toggleMenu() {
        if (this.isMenuOpen) {
            this.closeMenu();
        } else {
            this.openMenu();
        }
    }

    openMenu() {
        this.isMenuOpen = true;
        this.hamburgerMenu.classList.add('active');
        this.navMenu.classList.add('active');
        this.overlay.classList.add('active');
        
        // Prevent body scroll when menu is open
        document.body.style.overflow = 'hidden';
    }

    closeMenu() {
        this.isMenuOpen = false;
        this.hamburgerMenu.classList.remove('active');
        this.navMenu.classList.remove('active');
        this.overlay.classList.remove('active');
        
        // Restore body scroll
        document.body.style.overflow = '';
    }

    // Method to set active navigation item
    setActiveItem(href) {
        const navItems = this.navMenu.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === href) {
                item.classList.add('active');
            }
        });
    }
}

// Initialize navigation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.navigationController = new NavigationController();
    
    // Auto-set active item based on current page
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    window.navigationController.setActiveItem(currentPage);
});

// Expose for external use
window.NavigationController = NavigationController; 