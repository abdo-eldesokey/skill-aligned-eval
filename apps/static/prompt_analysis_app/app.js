// T2I Prompts Skill Analyzer - Frontend JavaScript

const API_BASE = 'http://localhost:5000/api';

let allPrompts = [];
let filteredPrompts = [];
let displayedPrompts = []; // Prompts shown in the list (after search)
let skillTaxonomy = {};
let selectedPromptId = null;
let selectedPromptIndex = -1;
let histogramChart = null;
let currentDirectory = '';
let currentJsonFile = '';

// Pagination
const PROMPTS_PER_PAGE = 50;
let currentPage = 1;
let totalPages = 1;

// Initialize the application
async function init() {
    initTheme();
    await loadDirectories();
    await loadJsonFiles();
    await loadSkillTaxonomy();
    await loadPrompts();
    await loadStatistics();
    setupEventListeners();
    updateHistogram('skills');
}

// Theme Management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    const body = document.body;
    const themeIcon = document.getElementById('themeIcon');
    
    if (theme === 'light') {
        body.setAttribute('data-theme', 'light');
        themeIcon.textContent = '☀️';
    } else {
        body.removeAttribute('data-theme');
        themeIcon.textContent = '🌙';
    }
    
    localStorage.setItem('theme', theme);
    
    // Update chart if it exists
    if (histogramChart) {
        updateChartTheme();
    }
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

function updateChartTheme() {
    const isDark = !document.body.hasAttribute('data-theme') || document.body.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(45, 59, 95, 0.3)' : 'rgba(203, 213, 225, 0.5)';
    
    if (histogramChart) {
        histogramChart.options.scales.x.ticks.color = textColor;
        histogramChart.options.scales.x.grid.color = gridColor;
        histogramChart.options.scales.y.ticks.color = textColor;
        histogramChart.options.scales.y.grid.color = gridColor;
        histogramChart.update();
    }
}

// Load available directories
async function loadDirectories() {
    try {
        const response = await fetch(`${API_BASE}/directories`);
        const data = await response.json();
        
        const select = document.getElementById('directorySelect');
        select.innerHTML = '';
        
        data.directories.forEach(dir => {
            const option = document.createElement('option');
            option.value = dir;
            option.textContent = dir;
            if (dir === data.default) {
                option.selected = true;
                currentDirectory = dir;
            }
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading directories:', error);
    }
}

// Load JSON files for current directory
async function loadJsonFiles() {
    try {
        const response = await fetch(`${API_BASE}/json_files?directory=${currentDirectory}`);
        const data = await response.json();
        
        const select = document.getElementById('jsonFileSelect');
        select.innerHTML = '';
        
        if (data.json_files.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No JSON files';
            select.appendChild(option);
            currentJsonFile = '';
            return;
        }
        
        data.json_files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            if (file === data.default) {
                option.selected = true;
                currentJsonFile = file;
            }
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading JSON files:', error);
    }
}

// Load skill taxonomy
async function loadSkillTaxonomy() {
    try {
        const response = await fetch(`${API_BASE}/skills`);
        skillTaxonomy = await response.json();
        populateSkillFilter();
    } catch (error) {
        console.error('Error loading skill taxonomy:', error);
    }
}

// Load all prompts
async function loadPrompts(skillFilter = '', subskillFilter = '') {
    try {
        let url = `${API_BASE}/prompts`;
        const params = new URLSearchParams();
        
        if (currentDirectory) params.append('directory', currentDirectory);
        if (currentJsonFile) params.append('json_file', currentJsonFile);
        if (skillFilter) params.append('skill', skillFilter);
        if (subskillFilter) params.append('subskill', subskillFilter);
        
        if (params.toString()) {
            url += `?${params.toString()}`;
        }
        
        const response = await fetch(url);
        const prompts = await response.json();
        
        allPrompts = prompts;
        filteredPrompts = prompts;
        displayedPrompts = prompts;
        
        // Apply search filter if there's a search term
        const searchTerm = document.getElementById('promptIdSearch')?.value || '';
        if (searchTerm.trim()) {
            applySearchFilter(searchTerm.trim());
        } else {
            renderPromptsList(displayedPrompts);
            updateStats();
        }
    } catch (error) {
        console.error('Error loading prompts:', error);
        document.getElementById('promptsList').innerHTML = 
            '<div class="loading">Error loading prompts</div>';
    }
}

// Load statistics
async function loadStatistics() {
    try {
        const params = new URLSearchParams();
        if (currentDirectory) params.append('directory', currentDirectory);
        if (currentJsonFile) params.append('json_file', currentJsonFile);
        const queryString = params.toString() ? `?${params.toString()}` : '';
        const response = await fetch(`${API_BASE}/statistics${queryString}`);
        const data = await response.json();
        
        document.getElementById('totalPrompts').textContent = data.total_prompts;
        document.getElementById('totalSkills').textContent = 
            Object.keys(skillTaxonomy).length;
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Populate skill filter dropdown
function populateSkillFilter() {
    const skillFilter = document.getElementById('skillFilter');
    
    Object.keys(skillTaxonomy).sort().forEach(skill => {
        const option = document.createElement('option');
        option.value = skill;
        option.textContent = skill.replace(/_/g, ' ');
        skillFilter.appendChild(option);
    });
}

// Populate subskill filter based on selected skill
function populateSubskillFilter(skill) {
    const subskillFilter = document.getElementById('subskillFilter');
    subskillFilter.innerHTML = '<option value="">All Subskills</option>';
    
    if (!skill || !skillTaxonomy[skill]) {
        subskillFilter.disabled = true;
        return;
    }
    
    const subskills = skillTaxonomy[skill];
    if (!subskills || subskills.length === 0) {
        subskillFilter.disabled = true;
        return;
    }
    
    subskillFilter.disabled = false;
    subskills.forEach(subskill => {
        const option = document.createElement('option');
        option.value = subskill;
        option.textContent = subskill.replace(/_/g, ' ');
        subskillFilter.appendChild(option);
    });
}

// Calculate pagination
function updatePagination(prompts) {
    totalPages = Math.ceil(prompts.length / PROMPTS_PER_PAGE);
    if (currentPage > totalPages) {
        currentPage = 1;
    }
    if (currentPage < 1) {
        currentPage = 1;
    }
}

// Get prompts for current page
function getPagePrompts(prompts) {
    const startIndex = (currentPage - 1) * PROMPTS_PER_PAGE;
    const endIndex = startIndex + PROMPTS_PER_PAGE;
    return prompts.slice(startIndex, endIndex);
}

// Render pagination controls
function renderPaginationControls() {
    const paginationContainerTop = document.getElementById('paginationControls');
    const paginationContainerBottom = document.getElementById('paginationControlsBottom');
    
    if (totalPages <= 1) {
        paginationContainerTop.innerHTML = '';
        paginationContainerBottom.innerHTML = '';
        return;
    }
    
    const startItem = (currentPage - 1) * PROMPTS_PER_PAGE + 1;
    const endItem = Math.min(currentPage * PROMPTS_PER_PAGE, displayedPrompts.length);
    
    let paginationHTML = `
        <div class="pagination">
            <button class="pagination-btn pagination-first" ${currentPage === 1 ? 'disabled' : ''}>
                ⟪ First
            </button>
            <button class="pagination-btn pagination-prev" ${currentPage === 1 ? 'disabled' : ''}>
                ‹ Prev
            </button>
            <span class="pagination-info">
                Page ${currentPage} of ${totalPages} (${startItem}-${endItem} of ${displayedPrompts.length})
            </span>
            <button class="pagination-btn pagination-next" ${currentPage === totalPages ? 'disabled' : ''}>
                Next ›
            </button>
            <button class="pagination-btn pagination-last" ${currentPage === totalPages ? 'disabled' : ''}>
                Last ⟫
            </button>
        </div>
    `;
    
    paginationContainerTop.innerHTML = paginationHTML;
    paginationContainerBottom.innerHTML = paginationHTML;
    
    // Add event listeners for both top and bottom pagination controls
    document.querySelectorAll('.pagination-first').forEach(btn => {
        btn.addEventListener('click', () => goToPage(1));
    });
    document.querySelectorAll('.pagination-prev').forEach(btn => {
        btn.addEventListener('click', () => goToPage(currentPage - 1));
    });
    document.querySelectorAll('.pagination-next').forEach(btn => {
        btn.addEventListener('click', () => goToPage(currentPage + 1));
    });
    document.querySelectorAll('.pagination-last').forEach(btn => {
        btn.addEventListener('click', () => goToPage(totalPages));
    });
}

// Go to specific page
function goToPage(page) {
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    renderPromptsList(filteredPrompts);
    renderPaginationControls();
    
    // Scroll to top of prompts list
    document.getElementById('promptsList').scrollTop = 0;
}

// Render prompts list with pagination
function renderPromptsList(prompts) {
    const promptsList = document.getElementById('promptsList');
    
    if (prompts.length === 0) {
        promptsList.innerHTML = '<div class="loading">No prompts found</div>';
        document.getElementById('paginationControls').innerHTML = '';
        document.getElementById('paginationControlsBottom').innerHTML = '';
        return;
    }
    
    displayedPrompts = prompts;
    
    // Update pagination
    updatePagination(prompts);
    const pagePrompts = getPagePrompts(prompts);
    const startIndex = (currentPage - 1) * PROMPTS_PER_PAGE;
    
    promptsList.innerHTML = pagePrompts.map((prompt, pageIndex) => {
        const globalIndex = startIndex + pageIndex;
        const annotations = prompt.annotations || [];
        // Get unique skills from annotations
        const uniqueSkills = [...new Set(annotations.map(ann => {
            const skill = ann.skill || '';
            const subskill = ann.subskill || '';
            return subskill ? `${skill}:${subskill}` : skill;
        }))];
        
        const skillTags = uniqueSkills.slice(0, 5).map(skill => 
            `<span class="skill-tag">${skill}</span>`
        ).join('');
        
        const moreSkills = uniqueSkills.length > 5 ? 
            `<span class="skill-tag">+${uniqueSkills.length - 5} more</span>` : '';
        
        return `
            <div class="prompt-card ${selectedPromptId === prompt.prompt_id ? 'selected' : ''}" 
                 data-prompt-id="${prompt.prompt_id}">
                <div class="prompt-card-header">
                    <span class="prompt-index">#${globalIndex + 1}</span>
                    <span class="prompt-id">ID: ${prompt.prompt_id}</span>
                </div>
                <div class="prompt-card-text">${escapeHtml(prompt.prompt)}</div>
                <div class="prompt-card-skills">
                    ${skillTags}
                    ${moreSkills}
                </div>
            </div>
        `;
    }).join('');
    
    // Add click listeners to prompt cards
    document.querySelectorAll('.prompt-card').forEach(card => {
        card.addEventListener('click', () => {
            const promptId = parseInt(card.dataset.promptId);
            selectPrompt(promptId);
        });
    });
    
    // Render pagination controls
    renderPaginationControls();
}

// Select and display a prompt
async function selectPrompt(promptId) {
    try {
        const params = new URLSearchParams();
        if (currentDirectory) params.append('directory', currentDirectory);
        if (currentJsonFile) params.append('json_file', currentJsonFile);
        const queryString = params.toString() ? `?${params.toString()}` : '';
        const response = await fetch(`${API_BASE}/prompts/${promptId}${queryString}`);
        const prompt = await response.json();
        
        if (prompt.error) {
            console.error('Prompt not found:', prompt.error);
            return;
        }
        
        selectedPromptId = promptId;
        
        // Find the index in filteredPrompts (all prompts, not just displayed)
        selectedPromptIndex = filteredPrompts.findIndex(p => p.prompt_id === promptId);
        
        renderPromptDetail(prompt);
        updateNavigationButtons();
        updatePromptCounter();
        
        // Update selected state in list
        document.querySelectorAll('.prompt-card').forEach(card => {
            card.classList.toggle('selected', 
                parseInt(card.dataset.promptId) === promptId);
        });
        
        // Show modal overlay
        document.getElementById('modalOverlay').style.display = 'flex';
    } catch (error) {
        console.error('Error loading prompt detail:', error);
    }
}

// Generate a color for each skill based on index
function getSkillColor(index) {
    const colors = [
        '#00d4ff', '#a855f7', '#10b981', '#f59e0b', '#ef4444',
        '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316', '#ec4899',
        '#3b82f6', '#6366f1', '#84cc16', '#eab308', '#f43f5e',
        '#0ea5e9', '#7c3aed', '#22c55e', '#fb923c', '#e11d48'
    ];
    return colors[index % colors.length];
}

// Highlight phrases in the prompt text using annotations
function highlightPromptText(promptText, annotations) {
    // Collect all phrase occurrences with their skills
    const allSegments = [];
    
    // Group annotations by unique skill:subskill combination for coloring
    const uniqueSkills = [];
    const skillColorMap = new Map();
    
    annotations.forEach((ann) => {
        const skill = ann.skill || '';
        const subskill = ann.subskill || '';
        const skillStr = subskill ? `${skill}:${subskill}` : skill;
        
        if (!skillColorMap.has(skillStr)) {
            const colorIndex = uniqueSkills.length;
            uniqueSkills.push(skillStr);
            skillColorMap.set(skillStr, { color: getSkillColor(colorIndex), index: colorIndex });
        }
    });
    
    annotations.forEach((ann) => {
        const skill = ann.skill || '';
        const subskill = ann.subskill || '';
        const skillStr = subskill ? `${skill}:${subskill}` : skill;
        const phrase = ann.phrase || '';
        
        if (!phrase) return;
        
        const skillInfo = skillColorMap.get(skillStr);
        const regex = new RegExp(escapeRegex(phrase), 'gi');
        let match;
        
        while ((match = regex.exec(promptText)) !== null) {
            const start = match.index;
            const end = start + phrase.length;
            
            allSegments.push({
                start,
                end,
                phrase,
                skill: { color: skillInfo.color, skillStr, index: skillInfo.index }
            });
        }
    });
    
    // Sort segments by start position, then by length (longest first)
    allSegments.sort((a, b) => {
        if (a.start !== b.start) return a.start - b.start;
        return (b.end - b.start) - (a.end - a.start);
    });
    
    // Merge overlapping segments and combine their skills
    const mergedSegments = [];
    
    for (const segment of allSegments) {
        // Check if this segment overlaps with any existing merged segment
        let merged = false;
        
        for (const existing of mergedSegments) {
            // Check for overlap
            if (segment.start < existing.end && segment.end > existing.start) {
                // Overlapping - merge them
                existing.start = Math.min(existing.start, segment.start);
                existing.end = Math.max(existing.end, segment.end);
                existing.phrase = promptText.substring(existing.start, existing.end);
                
                // Add skill if not already present
                const hasSkill = existing.skills.some(s => s.index === segment.skill.index);
                if (!hasSkill) {
                    existing.skills.push(segment.skill);
                }
                
                merged = true;
                break;
            }
        }
        
        if (!merged) {
            // No overlap - add as new segment
            mergedSegments.push({
                start: segment.start,
                end: segment.end,
                phrase: segment.phrase,
                skills: [segment.skill]
            });
        }
    }
    
    // Sort merged segments by position
    mergedSegments.sort((a, b) => a.start - b.start);
    
    // Build the highlighted HTML
    let result = '';
    let lastIndex = 0;
    
    mergedSegments.forEach(segment => {
        // Add text before this segment
        if (segment.start > lastIndex) {
            result += escapeHtml(promptText.substring(lastIndex, segment.start));
        }
        
        // Build skill labels and colors
        const skillLabels = segment.skills.map(s => s.skillStr.replace(/_/g, ' ')).join(' | ');
        const colors = segment.skills.map(s => s.color);
        
        // Create gradient or solid color background
        let backgroundStyle;
        let borderStyle;
        
        if (colors.length === 1) {
            backgroundStyle = `background-color: ${colors[0]}25;`;
            borderStyle = `border-bottom: 2px solid ${colors[0]};`;
        } else {
            // Multiple colors - create a gradient background and stacked border
            const gradientStops = colors.map((c, i) => {
                const start = (i / colors.length) * 100;
                const end = ((i + 1) / colors.length) * 100;
                return `${c}25 ${start}%, ${c}25 ${end}%`;
            }).join(', ');
            backgroundStyle = `background: linear-gradient(to right, ${gradientStops});`;
            
            // Create multi-color border using box-shadow
            const borderColors = colors.map((c, i) => 
                `inset 0 -${2 + i * 2}px 0 0 ${c}`
            ).join(', ');
            borderStyle = `box-shadow: ${borderColors};`;
        }
        
        // Add highlighted segment with click handler
        // Use the first skill if multiple skills are present
        const primarySkill = segment.skills[0].skillStr;
        result += `<mark class="highlight clickable-highlight" style="${backgroundStyle} ${borderStyle}" data-skill-label="${skillLabels}" data-skill="${escapeHtml(primarySkill)}">${escapeHtml(segment.phrase)}</mark>`;
        
        lastIndex = segment.end;
    });
    
    // Add remaining text
    if (lastIndex < promptText.length) {
        result += escapeHtml(promptText.substring(lastIndex));
    }
    
    return result;
}

// Render prompt detail with highlighted text
function renderPromptDetail(prompt) {
    document.getElementById('promptId').textContent = prompt.prompt_id;
    document.getElementById('datasetId').textContent = prompt.dataset_id || 'N/A';
    document.getElementById('promptType').textContent = prompt.prompt_type || 'N/A';
    document.getElementById('llmModel').textContent = prompt.llm_model || 'N/A';
    
    const annotations = prompt.annotations || [];
    
    // Render highlighted prompt text
    const promptTextElement = document.getElementById('promptText');
    promptTextElement.innerHTML = highlightPromptText(prompt.prompt, annotations);
    
    // Group annotations by unique skill:subskill for legend
    const uniqueSkills = [];
    const skillColorMap = new Map();
    
    annotations.forEach((ann) => {
        const skill = ann.skill || '';
        const subskill = ann.subskill || '';
        const skillStr = subskill ? `${skill}:${subskill}` : skill;
        
        if (!skillColorMap.has(skillStr)) {
            const colorIndex = uniqueSkills.length;
            uniqueSkills.push(skillStr);
            skillColorMap.set(skillStr, { color: getSkillColor(colorIndex), index: colorIndex });
        }
    });
    
    // Group annotations by skill for VQA display
    const annotationsBySkill = new Map();
    annotations.forEach((ann) => {
        const skill = ann.skill || '';
        const subskill = ann.subskill || '';
        const skillStr = subskill ? `${skill}:${subskill}` : skill;
        
        if (!annotationsBySkill.has(skillStr)) {
            annotationsBySkill.set(skillStr, []);
        }
        annotationsBySkill.get(skillStr).push(ann);
    });
    
    // Render legend with click handlers
    const legend = document.getElementById('skillLegend');
    legend.innerHTML = uniqueSkills.map((skillStr) => {
        const skillInfo = skillColorMap.get(skillStr);
        const color = skillInfo.color;
        const displayText = skillStr.replace(/_/g, ' ');
        
        return `
            <span class="legend-item clickable" data-skill="${escapeHtml(skillStr)}" style="border-color: ${color};">
                <span class="legend-color" style="background-color: ${color};"></span>
                <span class="legend-text">${displayText}</span>
            </span>
        `;
    }).join('');
    
    // Add click handlers to legend items
    document.querySelectorAll('.legend-item.clickable').forEach(item => {
        item.addEventListener('click', () => {
            const skillStr = item.dataset.skill;
            showVQAForSkill(skillStr, annotationsBySkill.get(skillStr), skillColorMap.get(skillStr).color);
        });
    });
    
    // Add click handlers to highlighted words in the prompt
    document.querySelectorAll('.highlight.clickable-highlight').forEach(highlight => {
        highlight.addEventListener('click', (e) => {
            e.stopPropagation();
            const skillStr = highlight.dataset.skill;
            if (skillStr && annotationsBySkill.has(skillStr)) {
                showVQAForSkill(skillStr, annotationsBySkill.get(skillStr), skillColorMap.get(skillStr).color);
            }
        });
    });
    
    // Initialize VQA container with placeholder
    const vqaContainer = document.getElementById('vqaQuestions');
    vqaContainer.innerHTML = `
        <div class="vqa-placeholder">
            <span class="vqa-placeholder-icon">💡</span>
            <p class="vqa-placeholder-text">Click on a skill in the legend above to view its VQA questions</p>
        </div>
    `;
    
    // Hide the old skill breakdown section
    document.getElementById('skillBreakdown').style.display = 'none';
}

// Show VQA questions for a specific skill
function showVQAForSkill(skillStr, annotations, color) {
    const vqaContainer = document.getElementById('vqaQuestions');
    const displayText = skillStr.replace(/_/g, ' ').replace(':', ': ');
    
    // Build a map of UIDs to annotations for dependency lookup
    const allAnnotations = selectedPromptId ? 
        filteredPrompts.find(p => p.prompt_id === selectedPromptId)?.annotations || [] : 
        [];
    const uidToAnnotation = new Map();
    allAnnotations.forEach(ann => {
        if (ann.uid) {
            uidToAnnotation.set(ann.uid, ann);
        }
    });
    
    vqaContainer.innerHTML = `
        <div class="vqa-skill-display">
            <div class="vqa-skill-display-header">
                <h3 class="vqa-title">
                    <span class="icon">❓</span>
                    VQA Questions for: <span style="color: ${color};">${displayText}</span>
                </h3>
                <button class="vqa-close-btn" onclick="closeVQADisplay()">✕</button>
            </div>
            <div class="vqa-skill-group" style="border-left-color: ${color};">
                <ul class="vqa-question-list">
                    ${annotations.map((ann) => {
                        const phrase = ann.phrase || '';
                        const question = ann.question || '';
                        const nodeType = ann.node_type || '';
                        const dependsOn = ann.depends_on || [];
                        
                        // Build dependency info
                        let dependencyInfo = '';
                        if (dependsOn && dependsOn.length > 0) {
                            const depQuestions = dependsOn
                                .map(uid => {
                                    const depAnn = uidToAnnotation.get(uid);
                                    return depAnn ? escapeHtml(depAnn.question) : uid;
                                })
                                .join(' • ');
                            // Create plain text version for tooltip
                            const depQuestionsPlain = dependsOn
                                .map(uid => {
                                    const depAnn = uidToAnnotation.get(uid);
                                    return depAnn ? depAnn.question : uid;
                                })
                                .join(' • ');
                            dependencyInfo = `
                                <div class="vqa-dependency" title="${escapeHtml(depQuestionsPlain)}">
                                    <span class="dependency-icon">🔗</span>
                                    <span class="dependency-text">${depQuestions}</span>
                                </div>
                            `;
                        }
                        
                        return `
                            <li class="vqa-question-item">
                                <div class="vqa-phrase" style="border-left-color: ${color}; background: linear-gradient(90deg, ${color}15, transparent);">
                                    <span class="phrase-icon">📝</span>
                                    <span class="phrase-text">"${escapeHtml(phrase)}"</span>
                                </div>
                                <div class="vqa-question">
                                    <span class="question-icon">❓</span>
                                    ${escapeHtml(question)}
                                    ${nodeType ? `<span class="node-type-badge">${nodeType}</span>` : ''}
                                </div>
                                ${dependencyInfo}
                            </li>
                        `;
                    }).join('')}
                </ul>
            </div>
        </div>
    `;
}

// Close VQA display and show placeholder
function closeVQADisplay() {
    const vqaContainer = document.getElementById('vqaQuestions');
    vqaContainer.innerHTML = `
        <div class="vqa-placeholder">
            <span class="vqa-placeholder-icon">💡</span>
            <p class="vqa-placeholder-text">Click on a skill in the legend above to view its VQA questions</p>
        </div>
    `;
}

// Update histogram
async function updateHistogram(type) {
    try {
        const params = new URLSearchParams();
        params.append('type', type);
        if (currentDirectory) params.append('directory', currentDirectory);
        if (currentJsonFile) params.append('json_file', currentJsonFile);
        
        const response = await fetch(`${API_BASE}/histogram?${params.toString()}`);
        const data = await response.json();
        
        renderHistogram(data.labels, data.counts, type);
    } catch (error) {
        console.error('Error loading histogram:', error);
    }
}

// Get color for a skill (parent skill determines the hue)
function getSkillHue(skillName, skillIndex) {
    // Extract parent skill name (before the colon if it exists)
    const parentSkill = skillName.split(':')[0];
    
    // Create a consistent hash for the parent skill
    const skillList = Object.keys(skillTaxonomy);
    let index = skillList.indexOf(parentSkill);
    
    // If not found in taxonomy, use the provided index
    if (index === -1) {
        index = skillIndex;
    }
    
    // Use golden angle for nice distribution
    return (index * 137.5) % 360;
}

// Render histogram using Chart.js - now shows ALL items with horizontal bars
function renderHistogram(labels, counts, type) {
    const ctx = document.getElementById('histogramChart');
    
    if (histogramChart) {
        histogramChart.destroy();
    }
    
    // Show ALL items (no limiting)
    const allLabels = labels.map(label => label.replace(/_/g, ' '));
    const allCounts = counts;
    
    // Theme-aware colors
    const isDark = !document.body.hasAttribute('data-theme') || document.body.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(45, 59, 95, 0.3)' : 'rgba(203, 213, 225, 0.5)';
    const tooltipBg = isDark ? 'rgba(26, 35, 71, 0.95)' : 'rgba(255, 255, 255, 0.95)';
    const tooltipBorder = isDark ? '#00d4ff' : '#0284c7';
    
    // Generate colors based on parent skill
    const barColors = labels.map((label, index) => {
        const hue = getSkillHue(label, index);
        
        // For subskills view, vary the lightness to distinguish subskills of same parent
        if (type === 'subskills' && label.includes(':')) {
            const subskillParts = label.split(':');
            const subskillIndex = subskillParts[1] ? subskillParts[1].charCodeAt(0) % 3 : 0;
            const lightnessVariation = [60, 55, 65][subskillIndex];
            return isDark 
                ? `hsla(${hue}, 70%, ${lightnessVariation}%, 0.7)`
                : `hsla(${hue}, 60%, ${lightnessVariation - 10}%, 0.7)`;
        }
        
        return isDark 
            ? `hsla(${hue}, 70%, 60%, 0.7)`
            : `hsla(${hue}, 60%, 50%, 0.7)`;
    });
    
    const borderColors = labels.map((label, index) => {
        const hue = getSkillHue(label, index);
        
        // For subskills view, vary the lightness to distinguish subskills of same parent
        if (type === 'subskills' && label.includes(':')) {
            const subskillParts = label.split(':');
            const subskillIndex = subskillParts[1] ? subskillParts[1].charCodeAt(0) % 3 : 0;
            const lightnessVariation = [60, 55, 65][subskillIndex];
            return isDark 
                ? `hsla(${hue}, 70%, ${lightnessVariation}%, 1)`
                : `hsla(${hue}, 60%, ${lightnessVariation - 10}%, 1)`;
        }
        
        return isDark 
            ? `hsla(${hue}, 70%, 60%, 1)`
            : `hsla(${hue}, 60%, 50%, 1)`;
    });
    
    // Calculate dynamic height based on number of items
    const chartContainer = document.querySelector('.chart-container');
    const minHeight = 400;
    const itemHeight = 35; // pixels per item
    const calculatedHeight = Math.max(minHeight, allLabels.length * itemHeight);
    chartContainer.style.height = `${calculatedHeight}px`;
    
    histogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: allLabels,
            datasets: [{
                label: type === 'skills' ? 'Occurrences' : 'Occurrences',
                data: allCounts,
                backgroundColor: barColors,
                borderColor: borderColors,
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal bars
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipBorder,
                    bodyColor: textColor,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `Occurrences: ${context.parsed.x}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        color: textColor,
                        font: {
                            size: 11
                        },
                        precision: 0
                    },
                    grid: {
                        color: gridColor
                    },
                    title: {
                        display: true,
                        text: 'Number of Occurrences',
                        color: textColor,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    ticks: {
                        color: textColor,
                        font: {
                            size: 11
                        },
                        autoSkip: false // Show all labels
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Update stats display
function updateStats() {
    document.getElementById('filteredPrompts').textContent = displayedPrompts.length;
}

// Close modal function
function closeModal() {
    document.getElementById('modalOverlay').style.display = 'none';
    selectedPromptId = null;
    selectedPromptIndex = -1;
    document.querySelectorAll('.prompt-card').forEach(card => {
        card.classList.remove('selected');
    });
}

// Update navigation button states
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevPrompt');
    const nextBtn = document.getElementById('nextPrompt');
    
    if (prevBtn && nextBtn) {
        prevBtn.disabled = selectedPromptIndex <= 0;
        nextBtn.disabled = selectedPromptIndex >= filteredPrompts.length - 1;
    }
}

// Update prompt counter display
function updatePromptCounter() {
    const currentElement = document.querySelector('.counter-current');
    const totalElement = document.querySelector('.counter-total');
    
    if (currentElement && totalElement) {
        currentElement.textContent = selectedPromptIndex + 1; // +1 for 1-based indexing
        totalElement.textContent = filteredPrompts.length;
    }
}

// Navigate to previous prompt
function navigateToPrevPrompt() {
    if (selectedPromptIndex > 0) {
        const prevPrompt = filteredPrompts[selectedPromptIndex - 1];
        selectPrompt(prevPrompt.prompt_id);
        
        // Scroll modal content to top
        const modalContent = document.querySelector('.modal-content');
        if (modalContent) {
            modalContent.scrollTop = 0;
        }
    }
}

// Navigate to next prompt
function navigateToNextPrompt() {
    if (selectedPromptIndex < filteredPrompts.length - 1) {
        const nextPrompt = filteredPrompts[selectedPromptIndex + 1];
        selectPrompt(nextPrompt.prompt_id);
        
        // Scroll modal content to top
        const modalContent = document.querySelector('.modal-content');
        if (modalContent) {
            modalContent.scrollTop = 0;
        }
    }
}

// Setup event listeners
function setupEventListeners() {
    // Theme toggle button
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    
    // Directory selection change
    document.getElementById('directorySelect').addEventListener('change', async (e) => {
        currentDirectory = e.target.value;
        // Reset filters and pagination
        document.getElementById('skillFilter').value = '';
        document.getElementById('subskillFilter').value = '';
        document.getElementById('subskillFilter').disabled = true;
        currentPage = 1;
        // Reload JSON files for new directory
        await loadJsonFiles();
        // Reload data
        await loadPrompts();
        await loadStatistics();
        const histType = document.querySelector('input[name="histType"]:checked')?.value || 'skills';
        await updateHistogram(histType);
        // Close modal if open
        closeModal();
    });
    
    // JSON file selection change
    document.getElementById('jsonFileSelect').addEventListener('change', async (e) => {
        currentJsonFile = e.target.value;
        // Reset filters and pagination
        document.getElementById('skillFilter').value = '';
        document.getElementById('subskillFilter').value = '';
        document.getElementById('subskillFilter').disabled = true;
        currentPage = 1;
        // Reload data
        await loadPrompts();
        await loadStatistics();
        const histType = document.querySelector('input[name="histType"]:checked')?.value || 'skills';
        await updateHistogram(histType);
        // Close modal if open
        closeModal();
    });
    
    // Skill filter change
    document.getElementById('skillFilter').addEventListener('change', (e) => {
        const skill = e.target.value;
        populateSubskillFilter(skill);
        applyFilters();
    });
    
    // Subskill filter change
    document.getElementById('subskillFilter').addEventListener('change', () => {
        applyFilters();
    });
    
    // Clear filters button
    document.getElementById('clearFilters').addEventListener('click', () => {
        document.getElementById('skillFilter').value = '';
        document.getElementById('subskillFilter').value = '';
        document.getElementById('subskillFilter').disabled = true;
        applyFilters();
    });
    
    // Close modal button
    document.getElementById('closeModal').addEventListener('click', closeModal);
    
    // Navigation buttons
    document.getElementById('prevPrompt').addEventListener('click', navigateToPrevPrompt);
    document.getElementById('nextPrompt').addEventListener('click', navigateToNextPrompt);
    
    // Close modal when clicking outside the modal container
    document.getElementById('modalOverlay').addEventListener('click', (e) => {
        if (e.target.id === 'modalOverlay') {
            closeModal();
        }
    });
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        const modalOpen = document.getElementById('modalOverlay').style.display === 'flex';
        
        if (modalOpen) {
            if (e.key === 'Escape') {
                closeModal();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                navigateToPrevPrompt();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                navigateToNextPrompt();
            }
        }
    });
    
    // Histogram type radio buttons
    document.querySelectorAll('input[name="histType"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateHistogram(e.target.value);
        });
    });
    
    // Search by prompt ID
    const searchInput = document.getElementById('promptIdSearch');
    searchInput.addEventListener('input', (e) => {
        searchByPromptId(e.target.value);
    });
    
    // Clear search button
    document.getElementById('clearSearch').addEventListener('click', () => {
        document.getElementById('promptIdSearch').value = '';
        displayedPrompts = filteredPrompts;
        currentPage = 1;
        renderPromptsList(displayedPrompts);
        updateStats();
    });
}

// Apply filters
function applyFilters() {
    const skill = document.getElementById('skillFilter').value;
    const subskill = document.getElementById('subskillFilter').value;
    
    // Reset to page 1 when filters change
    currentPage = 1;
    
    loadPrompts(skill, subskill);
}

// Search by prompt ID - filters displayed prompts
function searchByPromptId(searchTerm) {
    if (!searchTerm.trim()) {
        // If search is empty, show all prompts
        displayedPrompts = filteredPrompts;
    } else {
        applySearchFilter(searchTerm.trim());
    }
}

// Apply search filter to displayed prompts
function applySearchFilter(searchTerm) {
    const searchLower = searchTerm.toLowerCase();
    displayedPrompts = filteredPrompts.filter(prompt => {
        const idMatch = prompt.prompt_id.toString().toLowerCase().includes(searchLower);
        const textMatch = (prompt.prompt || '').toLowerCase().includes(searchLower);
        return idMatch || textMatch;
    });
    
    // Reset to page 1 when search changes
    currentPage = 1;
    
    // Re-render the list
    renderPromptsList(displayedPrompts);
    updateStats();
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Utility function to escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
