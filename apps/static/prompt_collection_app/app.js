// Global state
let allPrompts = [];
let filteredPrompts = [];
let selectedPrompts = new Set();
let skillTaxonomy = {};
let currentDirectory = '';
let currentJsonFile = '';

// API base URL
const API_BASE = '';

// Initialize the app
document.addEventListener('DOMContentLoaded', async () => {
    await loadDirectories();
    await loadSkills();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Directory and file selection
    document.getElementById('directorySelect').addEventListener('change', handleDirectoryChange);
    document.getElementById('jsonFileSelect').addEventListener('change', handleJsonFileChange);

    // Filters
    document.getElementById('skillFilter').addEventListener('change', handleSkillFilterChange);
    document.getElementById('subskillFilter').addEventListener('change', handleFilterChange);
    document.getElementById('showOnlySelected').addEventListener('change', handleFilterChange);
    document.getElementById('clearFilters').addEventListener('click', clearFilters);

    // Search
    document.getElementById('promptIdSearch').addEventListener('input', handleSearch);
    document.getElementById('clearSearch').addEventListener('click', clearSearch);

    // Save collection
    document.getElementById('saveCollection').addEventListener('click', openSaveModal);
    document.getElementById('closeSaveModal').addEventListener('click', closeSaveModal);
    document.getElementById('cancelSave').addEventListener('click', closeSaveModal);
    document.getElementById('confirmSave').addEventListener('click', saveCollection);

    // Load collection
    document.getElementById('loadCollection').addEventListener('click', loadCollection);
    document.getElementById('clearSelection').addEventListener('click', clearSelection);

    // Close modal on overlay click
    document.getElementById('saveModal').addEventListener('click', (e) => {
        if (e.target.id === 'saveModal') {
            closeSaveModal();
        }
    });
}

// Load available directories
async function loadDirectories() {
    try {
        const response = await fetch(`${API_BASE}/api/directories`);
        const data = await response.json();

        const select = document.getElementById('directorySelect');
        select.innerHTML = '';

        data.directories.forEach(dir => {
            const option = document.createElement('option');
            option.value = dir;
            option.textContent = dir;
            if (dir === data.default) {
                option.selected = true;
            }
            select.appendChild(option);
        });

        currentDirectory = data.default;
        await loadJsonFiles(currentDirectory);
    } catch (error) {
        console.error('Error loading directories:', error);
    }
}

// Load JSON files for a directory
async function loadJsonFiles(directory) {
    try {
        const response = await fetch(`${API_BASE}/api/json_files?directory=${directory}`);
        const data = await response.json();

        const select = document.getElementById('jsonFileSelect');
        select.innerHTML = '';

        const collectionSelect = document.getElementById('collectionSelect');
        collectionSelect.innerHTML = '<option value="">-- Select a collection --</option>';

        data.json_files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            if (file === data.default) {
                option.selected = true;
            }
            select.appendChild(option);

            // Add to collection dropdown as well
            const collectionOption = document.createElement('option');
            collectionOption.value = file;
            collectionOption.textContent = file;
            collectionSelect.appendChild(collectionOption);
        });

        currentJsonFile = data.default;
        await loadPrompts();
    } catch (error) {
        console.error('Error loading JSON files:', error);
    }
}

// Load skills taxonomy
async function loadSkills() {
    try {
        const response = await fetch(`${API_BASE}/api/skills`);
        skillTaxonomy = await response.json();
        populateSkillFilter();
    } catch (error) {
        console.error('Error loading skills:', error);
    }
}

// Populate skill filter dropdown
function populateSkillFilter() {
    const select = document.getElementById('skillFilter');
    select.innerHTML = '<option value="">All Skills</option>';

    Object.keys(skillTaxonomy).forEach(skill => {
        const option = document.createElement('option');
        option.value = skill;
        option.textContent = skill;
        select.appendChild(option);
    });
}

// Load prompts
async function loadPrompts() {
    try {
        const params = new URLSearchParams({
            directory: currentDirectory,
            json_file: currentJsonFile
        });

        const skillFilter = document.getElementById('skillFilter').value;
        const subskillFilter = document.getElementById('subskillFilter').value;

        if (skillFilter) params.append('skill', skillFilter);
        if (subskillFilter) params.append('subskill', subskillFilter);

        const response = await fetch(`${API_BASE}/api/prompts?${params}`);
        allPrompts = await response.json();

        updateStats();
        applyFilters();
    } catch (error) {
        console.error('Error loading prompts:', error);
        document.getElementById('promptsList').innerHTML =
            '<div class="empty-state"><div class="empty-state-icon">⚠️</div><p>Error loading prompts</p></div>';
    }
}

// Update statistics
function updateStats() {
    document.getElementById('totalPrompts').textContent = allPrompts.length;
    document.getElementById('selectedPrompts').textContent = selectedPrompts.size;
}

// Apply filters
function applyFilters() {
    const searchTerm = document.getElementById('promptIdSearch').value.toLowerCase();
    const showOnlySelected = document.getElementById('showOnlySelected').checked;

    filteredPrompts = allPrompts.filter(prompt => {
        // Search filter
        if (searchTerm && !prompt.prompt_id.toString().includes(searchTerm)) {
            return false;
        }

        // Show only selected filter
        if (showOnlySelected && !selectedPrompts.has(prompt.prompt_id)) {
            return false;
        }

        return true;
    });

    document.getElementById('filteredPrompts').textContent = filteredPrompts.length;
    renderPrompts();
}

// Render prompts
function renderPrompts() {
    const container = document.getElementById('promptsList');

    if (filteredPrompts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>No prompts found matching your filters</p>
            </div>
        `;
        return;
    }

    container.innerHTML = '';

    filteredPrompts.forEach(prompt => {
        const card = createPromptCard(prompt);
        container.appendChild(card);
    });
}

// Create prompt card
function createPromptCard(prompt) {
    const card = document.createElement('div');
    card.className = 'prompt-card';

    const isSelected = selectedPrompts.has(prompt.prompt_id);

    // Get unique skills
    const skills = new Set();
    if (prompt.annotations) {
        prompt.annotations.forEach(ann => {
            if (ann.skill) {
                if (ann.subskill) {
                    skills.add(`${ann.skill}: ${ann.subskill}`);
                } else {
                    skills.add(ann.skill);
                }
            }
        });
    }

    card.innerHTML = `
        <div class="prompt-content">
            <div class="prompt-header">
                <span class="prompt-id">ID: ${prompt.prompt_id}</span>
            </div>
            <div class="prompt-text">${escapeHtml(prompt.prompt)}</div>
            <div class="prompt-skills">
                ${Array.from(skills).map(skill =>
        `<span class="skill-tag">${escapeHtml(skill)}</span>`
    ).join('')}
            </div>
        </div>
        <button class="add-button ${isSelected ? 'selected' : ''}" data-prompt-id="${prompt.prompt_id}">
            ${isSelected ? '✓' : '+'}
        </button>
    `;

    // Add click handler to the button
    const button = card.querySelector('.add-button');
    button.addEventListener('click', () => togglePromptSelection(prompt.prompt_id, button));

    return card;
}

// Toggle prompt selection
function togglePromptSelection(promptId, button) {
    if (selectedPrompts.has(promptId)) {
        selectedPrompts.delete(promptId);
        button.classList.remove('selected');
        button.textContent = '+';
    } else {
        selectedPrompts.add(promptId);
        button.classList.add('selected');
        button.textContent = '✓';
    }

    updateStats();
}

// Handle directory change
async function handleDirectoryChange(e) {
    currentDirectory = e.target.value;
    await loadJsonFiles(currentDirectory);
}

// Handle JSON file change
async function handleJsonFileChange(e) {
    currentJsonFile = e.target.value;
    await loadPrompts();
}

// Handle skill filter change
function handleSkillFilterChange(e) {
    const skill = e.target.value;
    const subskillSelect = document.getElementById('subskillFilter');

    if (skill && skillTaxonomy[skill]) {
        subskillSelect.disabled = false;
        subskillSelect.innerHTML = '<option value="">All Subskills</option>';

        skillTaxonomy[skill].forEach(subskill => {
            const option = document.createElement('option');
            option.value = subskill;
            option.textContent = subskill;
            subskillSelect.appendChild(option);
        });
    } else {
        subskillSelect.disabled = true;
        subskillSelect.innerHTML = '<option value="">All Subskills</option>';
    }

    handleFilterChange();
}

// Handle filter change
async function handleFilterChange() {
    await loadPrompts();
}

// Handle search
function handleSearch() {
    applyFilters();
}

// Clear search
function clearSearch() {
    document.getElementById('promptIdSearch').value = '';
    applyFilters();
}

// Clear filters
function clearFilters() {
    document.getElementById('skillFilter').value = '';
    document.getElementById('subskillFilter').value = '';
    document.getElementById('subskillFilter').disabled = true;
    document.getElementById('showOnlySelected').checked = false;
    handleFilterChange();
}

// Open save modal
function openSaveModal() {
    if (selectedPrompts.size === 0) {
        alert('Please select at least one prompt to save.');
        return;
    }

    document.getElementById('savePromptCount').textContent = selectedPrompts.size;
    document.getElementById('collectionFilename').value = '';
    document.getElementById('saveStatus').style.display = 'none';
    document.getElementById('saveModal').style.display = 'flex';
}

// Close save modal
function closeSaveModal() {
    document.getElementById('saveModal').style.display = 'none';
}

// Save collection
async function saveCollection() {
    const filename = document.getElementById('collectionFilename').value.trim();

    if (!filename) {
        showSaveStatus('Please enter a filename', 'error');
        return;
    }

    // Get selected prompts data
    const selectedPromptsData = allPrompts.filter(p => selectedPrompts.has(p.prompt_id));

    try {
        const response = await fetch(`${API_BASE}/api/save_collection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename,
                prompts: selectedPromptsData,
                directory: currentDirectory
            })
        });

        const result = await response.json();

        if (response.ok) {
            showSaveStatus(`Successfully saved ${result.total_prompts} prompts to ${result.filepath}`, 'success');
            setTimeout(() => {
                closeSaveModal();
            }, 2000);
        } else {
            showSaveStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Error saving collection:', error);
        showSaveStatus('Error saving collection. Please try again.', 'error');
    }
}

// Show save status
function showSaveStatus(message, type) {
    const statusEl = document.getElementById('saveStatus');
    statusEl.textContent = message;
    statusEl.className = `save-status ${type}`;
    statusEl.style.display = 'block';
}

// Show load status
function showLoadStatus(message, type) {
    const statusEl = document.getElementById('loadStatus');
    statusEl.textContent = message;
    statusEl.className = `load-status ${type}`;
    statusEl.style.display = 'block';

    // Auto-hide after 3 seconds
    setTimeout(() => {
        statusEl.style.display = 'none';
    }, 3000);
}

// Load collection
async function loadCollection() {
    const filename = document.getElementById('collectionSelect').value;

    if (!filename) {
        showLoadStatus('Please select a collection file', 'error');
        return;
    }

    try {
        const params = new URLSearchParams({
            directory: currentDirectory,
            filename: filename
        });

        const response = await fetch(`${API_BASE}/api/load_collection?${params}`);
        const result = await response.json();

        if (response.ok) {
            // Add all prompt IDs from the collection to selectedPrompts
            result.prompt_ids.forEach(id => {
                selectedPrompts.add(id);
            });

            updateStats();
            renderPrompts();
            showLoadStatus(`Loaded ${result.total_prompts} prompts from ${filename}`, 'success');
        } else {
            showLoadStatus(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Error loading collection:', error);
        showLoadStatus('Error loading collection. Please try again.', 'error');
    }
}

// Clear selection
function clearSelection() {
    selectedPrompts.clear();
    updateStats();
    renderPrompts();
    showLoadStatus('Selection cleared', 'success');
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
