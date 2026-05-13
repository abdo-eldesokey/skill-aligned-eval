/**
 * Annotation Statistics Viewer - Frontend JavaScript
 */

// ==================== State ====================
const state = {
    currentTask: null,
    tasks: [],
    annotators: [],
    models: [],
    summaryData: null,
    images: [],
    currentImage: null,
    heatmapEnabled: false,
    heatmapEnabled: false,
    selectedAnnotators: new Set(),
    selectedFilterAnnotators: new Set(),
    selectedFilterModels: new Set()
};


// ==================== DOM Elements ====================
const elements = {
    taskModal: document.getElementById('task-modal'),
    taskSelect: document.getElementById('task-select'),
    startBtn: document.getElementById('start-btn'),
    changeTaskBtn: document.getElementById('change-task-btn'),
    taskNameBadge: document.getElementById('task-name-badge'),
    tabs: document.querySelectorAll('.tab'),
    summaryView: document.getElementById('summary-view'),
    perImageView: document.getElementById('per-image-view'),
    // Summary view elements
    totalImages: document.getElementById('total-images'),
    totalModels: document.getElementById('total-models'),
    totalAnnotators: document.getElementById('total-annotators'),
    agreementContent: document.getElementById('agreement-content'),
    modelTableBody: document.getElementById('model-table-body'),
    annotatorTableBody: document.getElementById('annotator-table-body'),
    // Per-image view elements
    imageSearch: document.getElementById('image-search'),
    imageList: document.getElementById('image-list'),
    mainImage: document.getElementById('main-image'),
    heatmapCanvas: document.getElementById('heatmap-canvas'),
    showHeatmap: document.getElementById('show-heatmap'),
    annotatorCheckboxes: document.getElementById('annotator-checkboxes'),
    infoModel: document.getElementById('info-model'),
    infoPromptId: document.getElementById('info-prompt-id'),
    likertSection: document.getElementById('likert-section'),
    likertMean: document.getElementById('likert-mean'),
    likertStd: document.getElementById('likert-std'),
    likertBreakdown: document.getElementById('likert-breakdown'),
    maskSection: document.getElementById('mask-section'),
    maskBreakdown: document.getElementById('mask-breakdown'),
    textSection: document.getElementById('text-section'),
    textBreakdown: document.getElementById('text-breakdown'),
    bqaSection: document.getElementById('bqa-section'),
    bqaBreakdown: document.getElementById('bqa-breakdown'),
    heatmapLegend: document.getElementById('heatmap-legend'),
    annotatorFilterList: document.getElementById('annotator-filter-list'),
    btnSelectAll: document.getElementById('btn-select-all'),
    btnDeselectAll: document.getElementById('btn-deselect-all'),
    modelFilterList: document.getElementById('model-filter-list'),
    btnSelectAllModels: document.getElementById('btn-select-all-models'),
    btnDeselectAllModels: document.getElementById('btn-deselect-all-models')
};



// ==================== API Functions ====================
async function fetchTasks() {
    const response = await fetch('/api/tasks');
    const data = await response.json();
    return data.tasks;
}

async function fetchAnnotators() {
    const response = await fetch('/api/annotators');
    const data = await response.json();
    return data.annotators;
}

async function fetchTaskSummary(taskId, annotators = null, models = null) {
    let url = `/api/task/${taskId}/summary`;
    const params = [];
    if (annotators !== null) {
        params.push(`annotators=${annotators.join(',')}`);
    }
    if (models !== null) {
        params.push(`models=${models.join(',')}`);
    }
    if (params.length > 0) {
        url += `?${params.join('&')}`;
    }
    const response = await fetch(url);
    return response.json();
}


async function fetchTaskModels(taskId) {
    const response = await fetch(`/api/task/${taskId}/models`);
    const data = await response.json();
    return data.models;
}

async function fetchTaskImages(taskId) {
    const response = await fetch(`/api/task/${taskId}/images`);
    const data = await response.json();
    return data.images;
}

async function fetchImageStats(taskId, model, promptId) {
    const response = await fetch(`/api/task/${taskId}/image/${model}/${promptId}`);
    return response.json();
}

function getHeatmapUrl(taskId, model, promptId, annotators = null) {
    let url = `/api/task/${taskId}/heatmap/${model}/${promptId}`;
    if (annotators && annotators.length > 0) {
        url += `?annotators=${annotators.join(',')}`;
    }
    return url;
}

function getImageUrl(model, promptId, datasetVersion) {
    return `/assets/images/${datasetVersion}/${promptId}_${model}.png`;
}

// ==================== UI Functions ====================
function showModal() {
    elements.taskModal.classList.add('active');
}

function hideModal() {
    elements.taskModal.classList.remove('active');
}

function populateTaskSelect(tasks) {
    elements.taskSelect.innerHTML = '';
    tasks.forEach(task => {
        const option = document.createElement('option');
        option.value = task.id;
        option.textContent = task.name;
        elements.taskSelect.appendChild(option);
    });
}

function switchView(viewName) {
    elements.tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === viewName);
    });
    elements.summaryView.classList.toggle('active', viewName === 'summary');
    elements.perImageView.classList.toggle('active', viewName === 'per-image');
}

function getAgreementClass(alpha) {
    if (alpha >= 0.8) return 'good';
    if (alpha >= 0.6) return 'moderate';
    return 'poor';
}

function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return '-';
    return Number(num).toFixed(decimals);
}

function renderSummaryView(data) {
    // Overview stats
    elements.totalImages.textContent = data.total_images || 0;
    elements.totalModels.textContent = data.models?.length || 0;
    elements.totalAnnotators.textContent = data.annotators?.length || 0;

    // Agreement section - extract Krippendorff's Alpha
    elements.agreementContent.innerHTML = '';
    const modelStats = data.statistics?.models || {};

    // Track if we've shown any per-model agreement
    let hasPerModelAgreement = false;

    for (const [model, stats] of Object.entries(modelStats)) {
        // Check for Likert agreement
        if (stats.likert?.krippendorff_alpha !== undefined) {
            const alpha = stats.likert.krippendorff_alpha;
            const div = document.createElement('div');
            div.className = 'agreement-item';

            let html = `
                <span class="agreement-label">${model} (Rating)</span>
                <span class="agreement-value ${getAgreementClass(alpha)}">
                    α = ${formatNumber(alpha, 3)}
                </span>
            `;

            if (stats.likert.total_unsure_rate !== undefined) {
                const unsure = formatNumber(stats.likert.total_unsure_rate, 1);
                html += `
                <span class="agreement-value" style="font-size: 0.85em; opacity: 0.8; margin-top: 2px;">
                    Unsure: ${unsure}%
                </span>
                `;
            }

            div.innerHTML = html;
            elements.agreementContent.appendChild(div);
            hasPerModelAgreement = true;
        }

        // Check for text_per_word agreement
        if (stats.text_per_word?.krippendorff_alpha !== undefined) {
            const alpha = stats.text_per_word.krippendorff_alpha;
            const alphaRaw = stats.text_per_word.krippendorff_alpha_raw;

            const div = document.createElement('div');
            div.className = 'agreement-item';

            let html = `
                <span class="agreement-label">${model} (Text Accuracy)</span>
                <span class="agreement-value ${getAgreementClass(alpha)}">
                    Net α = ${formatNumber(alpha, 3)}
                </span>
            `;

            if (alphaRaw !== undefined) {
                html += `
                <span class="agreement-value ${getAgreementClass(alphaRaw)}" style="font-size: 0.85em; opacity: 0.8; margin-top: 2px;">
                    Raw α = ${formatNumber(alphaRaw, 3)}
                </span>
                `;
            }

            div.innerHTML = html;
            elements.agreementContent.appendChild(div);
            hasPerModelAgreement = true;
        }
    }

    // Check for Artifact Mask agreement
    const artifactStats = data.statistics?.artifact_mask || {};
    for (const [model, stats] of Object.entries(artifactStats)) {
        // Skip the average field (it's not a model)
        if (model === 'average_krippendorff_alpha') continue;

        if (stats.krippendorff_alpha !== undefined) {
            const alpha = stats.krippendorff_alpha;
            const div = document.createElement('div');
            div.className = 'agreement-item';
            div.innerHTML = `
                <span class="agreement-label">${model} (Artifacts)</span>
                <span class="agreement-value ${getAgreementClass(alpha)}">
                    α = ${formatNumber(alpha, 3)}
                </span>
            `;
            elements.agreementContent.appendChild(div);
            hasPerModelAgreement = true;
        }
    }

    // Check for Artifact Point agreement
    const pointStats = data.statistics?.artifact_point || {};
    for (const [model, stats] of Object.entries(pointStats)) {
        if (stats.mean_agreement_score !== undefined && stats.mean_agreement_score !== null) {
            const agreement = stats.mean_agreement_score;
            const div = document.createElement('div');
            div.className = 'agreement-item';
            div.innerHTML = `
                <span class="agreement-label">${model} (Points)</span>
                <span class="agreement-value ${getAgreementClass(agreement)}">
                    F1 = ${formatNumber(agreement, 3)}
                </span>
            `;
            elements.agreementContent.appendChild(div);
            hasPerModelAgreement = true;
        }
    }

    // Check for BQA agreement
    const bqaStats = data.statistics?.bqa || {};
    for (const [model, stats] of Object.entries(bqaStats)) {
        if (stats.krippendorff_alpha !== undefined) {
            const alpha = stats.krippendorff_alpha;
            const uncertaintyRate = stats.uncertainty_rate || 0;
            const div = document.createElement('div');
            div.className = 'agreement-item';
            div.innerHTML = `
                <span class="agreement-label">${model} (BQA)</span>
                <span class="agreement-value ${getAgreementClass(alpha)}">
                    α = ${formatNumber(alpha, 3)}
                </span>
                <span class="agreement-value" style="font-size: 0.85em; opacity: 0.8; margin-top: 2px;">
                    Unsure: ${formatNumber(uncertaintyRate, 1)}%
                </span>
            `;
            elements.agreementContent.appendChild(div);
            hasPerModelAgreement = true;
        }
    }

    // Add separator before averages if we have per-model data
    if (hasPerModelAgreement) {
        const separator = document.createElement('div');
        separator.style.borderTop = '2px solid #ddd';
        separator.style.marginTop = '8px';
        separator.style.marginBottom = '8px';
        elements.agreementContent.appendChild(separator);
    }

    // Display average agreements across models
    if (data.statistics?.average_likert_alpha !== undefined) {
        const avgAlpha = data.statistics.average_likert_alpha;
        const div = document.createElement('div');
        div.className = 'agreement-item';
        div.innerHTML = `
            <span class="agreement-label"><strong>Average (Rating)</strong></span>
            <span class="agreement-value ${getAgreementClass(avgAlpha)}">
                <strong>α = ${formatNumber(avgAlpha, 3)}</strong>
            </span>
        `;
        elements.agreementContent.appendChild(div);
    }

    if (data.statistics?.average_point_agreement !== undefined) {
        const avgAgreement = data.statistics.average_point_agreement;
        const div = document.createElement('div');
        div.className = 'agreement-item';
        div.innerHTML = `
            <span class="agreement-label"><strong>Average (Points)</strong></span>
            <span class="agreement-value ${getAgreementClass(avgAgreement)}">
                <strong>F1 = ${formatNumber(avgAgreement, 3)}</strong>
            </span>
        `;
        elements.agreementContent.appendChild(div);
    }

    if (data.statistics?.average_text_per_word_alpha !== undefined) {
        const avgAlpha = data.statistics.average_text_per_word_alpha;
        const div = document.createElement('div');
        div.className = 'agreement-item';
        div.innerHTML = `
            <span class="agreement-label"><strong>Average (Text Accuracy)</strong></span>
            <span class="agreement-value ${getAgreementClass(avgAlpha)}">
                <strong>α = ${formatNumber(avgAlpha, 3)}</strong>
            </span>
        `;
        elements.agreementContent.appendChild(div);
    }

    if (artifactStats.average_krippendorff_alpha !== undefined) {
        const avgAlpha = artifactStats.average_krippendorff_alpha;
        const div = document.createElement('div');
        div.className = 'agreement-item';
        div.innerHTML = `
            <span class="agreement-label"><strong>Average (Artifacts)</strong></span>
            <span class="agreement-value ${getAgreementClass(avgAlpha)}">
                <strong>α = ${formatNumber(avgAlpha, 3)}</strong>
            </span>
        `;
        elements.agreementContent.appendChild(div);
    }

    if (data.statistics?.average_bqa_alpha !== undefined) {
        const avgAlpha = data.statistics.average_bqa_alpha;
        const div = document.createElement('div');
        div.className = 'agreement-item';
        div.innerHTML = `
            <span class="agreement-label"><strong>Average (BQA)</strong></span>
            <span class="agreement-value ${getAgreementClass(avgAlpha)}">
                <strong>α = ${formatNumber(avgAlpha, 3)}</strong>
            </span>
        `;
        elements.agreementContent.appendChild(div);
    }

    if (elements.agreementContent.children.length === 0) {
        elements.agreementContent.innerHTML = '<p class="text-muted">No agreement data available</p>';
    }

    // Model comparison table
    elements.modelTableBody.innerHTML = '';
    for (const [model, stats] of Object.entries(modelStats)) {
        const row = document.createElement('tr');
        let score = '-';

        if (stats.likert?.mean !== undefined) {
            score = formatNumber(stats.likert.mean, 2) + ' ★';
            if (stats.likert.total_unsure_rate !== undefined) {
                const unsure = formatNumber(stats.likert.total_unsure_rate, 1);
                score += ` <small class="text-muted">(Unsure: ${unsure}%)</small>`;
            }
        } else if (stats.text_per_word?.mean_accuracy !== undefined) {
            const net = formatNumber(stats.text_per_word.mean_accuracy, 1);
            score = `${net}%`;
            if (stats.text_per_word.mean_accuracy_raw !== undefined) {
                const raw = formatNumber(stats.text_per_word.mean_accuracy_raw, 1);
                score += ` <small class="text-muted">(Raw: ${raw}%)</small>`;
            }
        } else if (stats.bqa?.yes_rate !== undefined) {
            score = `${formatNumber(stats.bqa.yes_rate, 1)}%`;
            if (stats.bqa.uncertainty_rate !== undefined) {
                const uncertainty = formatNumber(stats.bqa.uncertainty_rate, 1);
                score += ` <small class="text-muted">(Unsure: ${uncertainty}%)</small>`;
            }
        } else if (stats.artifact_point?.mean_points_per_image !== undefined) {
            score = `${formatNumber(stats.artifact_point.mean_points_per_image, 1)} pts`;
        }

        row.innerHTML = `
            <td>${model}</td>
            <td>${score}</td>
            <td>${stats.prompt_count || 0}</td>
        `;
        elements.modelTableBody.appendChild(row);
    }

    // Per-annotator statistics
    elements.annotatorTableBody.innerHTML = '';
    const annotatorAggregated = {};

    for (const [model, stats] of Object.entries(modelStats)) {
        const perAnnotator = stats.likert?.per_annotator || stats.text_per_word?.per_annotator || stats.artifact_point?.per_annotator || {};
        for (const [annotator, annStats] of Object.entries(perAnnotator)) {
            if (!annotatorAggregated[annotator]) {
                annotatorAggregated[annotator] = { scores: [], scoresRaw: [], count: 0, points: [] };
            }
            if (annStats.mean !== undefined) {
                annotatorAggregated[annotator].scores.push(annStats.mean);
            } else if (annStats.mean_accuracy !== undefined) {
                annotatorAggregated[annotator].scores.push(annStats.mean_accuracy);
                if (annStats.mean_accuracy_raw !== undefined) {
                    annotatorAggregated[annotator].scoresRaw.push(annStats.mean_accuracy_raw);
                }
            } else if (annStats.mean_points !== undefined) {
                annotatorAggregated[annotator].points.push(annStats.mean_points);
            }
            annotatorAggregated[annotator].count += annStats.count || 1;
        }
    }

    for (const [annotator, aggStats] of Object.entries(annotatorAggregated)) {
        const row = document.createElement('tr');
        let scoreDisplay = '-';

        if (aggStats.points.length > 0) {
            const avgPoints = aggStats.points.reduce((a, b) => a + b) / aggStats.points.length;
            scoreDisplay = formatNumber(avgPoints, 1) + ' pts';
        } else {
            const avgScore = aggStats.scores.length > 0
                ? aggStats.scores.reduce((a, b) => a + b) / aggStats.scores.length
                : null;

            scoreDisplay = formatNumber(avgScore, 2);

            if (aggStats.scoresRaw && aggStats.scoresRaw.length > 0) {
                const avgRaw = aggStats.scoresRaw.reduce((a, b) => a + b) / aggStats.scoresRaw.length;
                scoreDisplay += ` <small class="text-muted">(${formatNumber(avgRaw, 1)}%)</small>`;
            }
        }

        row.innerHTML = `
            <td>${annotator}</td>
            <td>${scoreDisplay}</td>
            <td>${aggStats.count}</td>
        `;
        elements.annotatorTableBody.appendChild(row);
    }
}

function renderImageList(images) {
    elements.imageList.innerHTML = '';

    images.forEach((img, index) => {
        const div = document.createElement('div');
        div.className = 'image-item';
        div.dataset.index = index;
        div.innerHTML = `
            <div class="image-item-id">${img.prompt_id}</div>
            <div class="image-item-model">${img.model}</div>
        `;
        div.addEventListener('click', () => selectImage(index));
        elements.imageList.appendChild(div);
    });
}

async function selectImage(index) {
    if (index < 0 || index >= state.images.length) return;

    state.currentImage = state.images[index];

    // Update active state in list
    document.querySelectorAll('.image-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });

    // Load image - use image_path from backend if available (supports WebP)
    const task = state.currentTask;
    const datasetVersion = task.dataset_version;
    const imageUrl = state.currentImage.image_path
        ? `/assets/images/${state.currentImage.image_path}`
        : getImageUrl(state.currentImage.model, state.currentImage.prompt_id, datasetVersion);
    elements.mainImage.src = imageUrl;

    // Update info panel
    elements.infoModel.textContent = state.currentImage.model;
    elements.infoPromptId.textContent = state.currentImage.prompt_id;

    // Fetch and render stats
    const stats = await fetchImageStats(task.id, state.currentImage.model, state.currentImage.prompt_id);
    renderImageStats(stats);

    // Setup annotator checkboxes for heatmap
    setupAnnotatorCheckboxes(stats);

    // Update heatmap if enabled
    if (state.heatmapEnabled) {
        loadHeatmap();
    }
}

function renderImageStats(stats) {
    // Likert section
    if (stats.statistics?.likert) {
        const likert = stats.statistics.likert;
        elements.likertSection.style.display = 'block';
        elements.likertMean.textContent = formatNumber(likert.mean, 2) + ' ★';
        elements.likertStd.textContent = formatNumber(likert.std, 2);

        elements.likertBreakdown.innerHTML = '';
        for (const [annotator, value] of Object.entries(likert.annotator_values || {})) {
            const div = document.createElement('div');
            div.className = 'annotator-stat';

            // Build raw details string
            let detailsHtml = '';
            if (stats.annotations && stats.annotations[annotator]) {
                const raws = Object.entries(stats.annotations[annotator])
                    .filter(([k, v]) => !k.startsWith('_'))
                    .map(([k, v]) => `<div><small><strong>${k}:</strong> ${v}</small></div>`)
                    .join('');
                if (raws) detailsHtml = `<div class="raw-details" style="margin-top: 5px;">${raws}</div>`;
            }

            div.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span>${annotator}</span>
                    <span class="annotator-stat-value">${value} ★</span>
                </div>
                ${detailsHtml}
            `;
            elements.likertBreakdown.appendChild(div);
        }
    } else {
        elements.likertSection.style.display = 'none';
    }

    // Mask section
    if (stats.statistics?.artifact_masks && Object.keys(stats.statistics.artifact_masks).length > 0) {
        elements.maskSection.style.display = 'block';
        elements.maskBreakdown.innerHTML = '';

        for (const [annotator, maskStats] of Object.entries(stats.statistics.artifact_masks)) {
            const div = document.createElement('div');
            div.className = 'annotator-stat';

            // Build raw details string
            let detailsHtml = '';
            if (stats.annotations && stats.annotations[annotator]) {
                const raws = Object.entries(stats.annotations[annotator])
                    .filter(([k, v]) => !k.startsWith('_'))
                    .map(([k, v]) => `<div><small><strong>${k}:</strong> ${v}</small></div>`)
                    .join('');
                if (raws) detailsHtml = `<div class="raw-details" style="margin-top: 5px;">${raws}</div>`;
            }

            div.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span>${annotator}</span>
                    <span class="annotator-stat-value">${formatNumber(maskStats.score, 1)}%</span>
                </div>
                ${detailsHtml}
            `;
            elements.maskBreakdown.appendChild(div);
        }
    } else {
        elements.maskSection.style.display = 'none';
    }

    // Text per-word section
    if (stats.statistics?.text_per_word && Object.keys(stats.statistics.text_per_word).length > 0) {
        elements.textSection.style.display = 'block';
        elements.textBreakdown.innerHTML = '';

        for (const [annotator, textStats] of Object.entries(stats.statistics.text_per_word)) {
            const div = document.createElement('div');
            div.className = 'annotator-stat';

            // Build raw details string
            let detailsHtml = '';
            if (stats.annotations && stats.annotations[annotator]) {
                const raws = Object.entries(stats.annotations[annotator])
                    .filter(([k, v]) => !k.startsWith('_'))
                    .map(([k, v]) => `<div><small><strong>${k}:</strong> ${v}</small></div>`)
                    .join('');
                if (raws) detailsHtml = `<div class="raw-details" style="margin-top: 5px;">${raws}</div>`;
            }

            div.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span>${annotator}</span>
                    <span class="annotator-stat-value">
                        Net: ${formatNumber(textStats.word_accuracy, 1)}% 
                        / Raw: ${formatNumber(textStats.word_accuracy_raw, 1)}%
                        <small>(${textStats.text_artifacts_penalty || 0} pen)</small>
                    </span>
                </div>
                ${detailsHtml}
            `;
            elements.textBreakdown.appendChild(div);
        }
    } else {
        elements.textSection.style.display = 'none';
    }

    // BQA section
    if (stats.statistics?.bqa && Object.keys(stats.statistics.bqa).length > 0) {
        elements.bqaSection.style.display = 'block';
        elements.bqaBreakdown.innerHTML = '';

        for (const [annotator, bqaStats] of Object.entries(stats.statistics.bqa)) {
            const div = document.createElement('div');
            div.className = 'annotator-stat';

            // Build raw details string
            let detailsHtml = '';
            if (stats.annotations && stats.annotations[annotator]) {
                const raws = Object.entries(stats.annotations[annotator])
                    .filter(([k, v]) => !k.startsWith('_'))
                    .map(([k, v]) => `<div><small><strong>${k}:</strong> ${v}</small></div>`)
                    .join('');
                if (raws) detailsHtml = `<div class="raw-details" style="margin-top: 5px;">${raws}</div>`;
            }

            let summary = `Y: ${bqaStats.yes_count} | N: ${bqaStats.no_count}`;
            if (bqaStats.unsure_count > 0) summary += ` | ?: ${bqaStats.unsure_count}`;

            div.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span>${annotator}</span>
                    <span class="annotator-stat-value">${summary}</span>
                </div>
                ${detailsHtml}
            `;
            elements.bqaBreakdown.appendChild(div);
        }
    } else {
        elements.bqaSection.style.display = 'none';
    }
}

function setupAnnotatorCheckboxes(stats) {
    elements.annotatorCheckboxes.innerHTML = '';
    state.selectedAnnotators.clear();

    const masks = stats.statistics?.artifact_masks || {};
    const annotators = Object.keys(masks);

    if (annotators.length === 0) {
        elements.showHeatmap.disabled = true;
        return;
    }

    elements.showHeatmap.disabled = false;

    annotators.forEach(annotator => {
        state.selectedAnnotators.add(annotator);

        const label = document.createElement('label');
        label.className = 'annotator-checkbox';
        label.innerHTML = `
            <input type="checkbox" checked data-annotator="${annotator}">
            ${annotator}
        `;

        const checkbox = label.querySelector('input');
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                state.selectedAnnotators.add(annotator);
            } else {
                state.selectedAnnotators.delete(annotator);
            }
            if (state.heatmapEnabled) {
                loadHeatmap();
            }
        });

        elements.annotatorCheckboxes.appendChild(label);
    });
}

function renderAnnotatorFilter() {
    elements.annotatorFilterList.innerHTML = '';
    state.annotators.forEach(annotator => {
        const label = document.createElement('label');
        label.className = 'filter-checkbox';
        label.innerHTML = `
            <input type="checkbox" value="${annotator}" ${state.selectedFilterAnnotators.has(annotator) ? 'checked' : ''}>
            ${annotator}
        `;
        const checkbox = label.querySelector('input');
        checkbox.addEventListener('change', async (e) => {
            if (e.target.checked) {
                state.selectedFilterAnnotators.add(annotator);
            } else {
                state.selectedFilterAnnotators.delete(annotator);
            }
            await reloadTaskSummary();
        });
        elements.annotatorFilterList.appendChild(label);
    });
}

function renderModelFilter() {
    elements.modelFilterList.innerHTML = '';
    state.models.forEach(model => {
        const label = document.createElement('label');
        label.className = 'filter-checkbox';
        label.innerHTML = `
            <input type="checkbox" value="${model}" ${state.selectedFilterModels.has(model) ? 'checked' : ''}>
            ${model}
        `;
        const checkbox = label.querySelector('input');
        checkbox.addEventListener('change', async (e) => {
            if (e.target.checked) {
                state.selectedFilterModels.add(model);
            } else {
                state.selectedFilterModels.delete(model);
            }
            await reloadTaskSummary();
        });
        elements.modelFilterList.appendChild(label);
    });
}

async function loadHeatmap() {
    if (!state.currentImage || !state.currentTask) return;

    const task = state.currentTask;
    const annotators = Array.from(state.selectedAnnotators);

    if (annotators.length === 0) {
        elements.heatmapCanvas.classList.remove('visible');
        elements.heatmapLegend.classList.remove('visible');
        return;
    }


    const url = getHeatmapUrl(task.id, state.currentImage.model, state.currentImage.prompt_id, annotators);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            elements.heatmapCanvas.classList.remove('visible');
            elements.heatmapLegend.classList.remove('visible');
            return;
        }


        const blob = await response.blob();
        const img = new Image();
        img.onload = () => {
            const canvas = elements.heatmapCanvas;
            const ctx = canvas.getContext('2d');

            // Match canvas size to displayed image
            const mainImg = elements.mainImage;
            canvas.width = mainImg.naturalWidth;
            canvas.height = mainImg.naturalHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            canvas.classList.add('visible');
            elements.heatmapLegend.classList.add('visible');
        };
        img.src = URL.createObjectURL(blob);
    } catch (error) {
        console.error('Error loading heatmap:', error);
        elements.heatmapCanvas.classList.remove('visible');
        elements.heatmapLegend.classList.remove('visible');
    }
}


function filterImages(searchTerm) {
    const items = document.querySelectorAll('.image-item');
    const term = searchTerm.toLowerCase();

    items.forEach(item => {
        const promptId = item.querySelector('.image-item-id').textContent.toLowerCase();
        const model = item.querySelector('.image-item-model').textContent.toLowerCase();
        const matches = promptId.includes(term) || model.includes(term);
        item.style.display = matches ? 'block' : 'none';
    });
}

// ==================== Event Handlers ====================
async function handleStartClick() {
    const selectedTaskId = elements.taskSelect.value;
    if (!selectedTaskId) return;

    state.currentTask = state.tasks.find(t => t.id === selectedTaskId);
    elements.taskNameBadge.textContent = state.currentTask.name;

    hideModal();

    // Load models for this task
    state.models = await fetchTaskModels(selectedTaskId);
    state.selectedFilterModels = new Set(state.models);
    renderModelFilter();

    // Load data
    state.summaryData = await fetchTaskSummary(selectedTaskId, Array.from(state.selectedFilterAnnotators), Array.from(state.selectedFilterModels));
    state.images = await fetchTaskImages(selectedTaskId);


    // Render views
    renderSummaryView(state.summaryData);
    renderImageList(state.images);

    // Select first image if available
    if (state.images.length > 0) {
        selectImage(0);
    }
}

function handleChangeTaskClick() {
    showModal();
}

function handleTabClick(e) {
    const viewName = e.target.dataset.view;
    if (viewName) {
        switchView(viewName);
    }
}

function handleHeatmapToggle(e) {
    state.heatmapEnabled = e.target.checked;
    if (state.heatmapEnabled) {
        loadHeatmap();
    } else {
        elements.heatmapCanvas.classList.remove('visible');
        elements.heatmapLegend.classList.remove('visible');
    }
}


function handleImageSearch(e) {
    filterImages(e.target.value);
}

async function reloadTaskSummary() {
    if (!state.currentTask) return;
    const annotators = Array.from(state.selectedFilterAnnotators);
    const models = Array.from(state.selectedFilterModels);
    state.summaryData = await fetchTaskSummary(state.currentTask.id, annotators, models);
    renderSummaryView(state.summaryData);
}

function handleSelectAll() {
    state.annotators.forEach(a => state.selectedFilterAnnotators.add(a));
    renderAnnotatorFilter();
    reloadTaskSummary();
}

function handleDeselectAll() {
    state.selectedFilterAnnotators.clear();
    renderAnnotatorFilter();
    reloadTaskSummary();
}

function handleSelectAllModels() {
    state.models.forEach(m => state.selectedFilterModels.add(m));
    renderModelFilter();
    reloadTaskSummary();
}

function handleDeselectAllModels() {
    state.selectedFilterModels.clear();
    renderModelFilter();
    reloadTaskSummary();
}

// ==================== Initialization ====================
async function init() {
    // Fetch initial data
    state.tasks = await fetchTasks();
    state.annotators = await fetchAnnotators();

    // Initialize filter set
    state.selectedFilterAnnotators = new Set(state.annotators);
    renderAnnotatorFilter();


    // Populate task select
    populateTaskSelect(state.tasks);

    // Setup event listeners
    elements.startBtn.addEventListener('click', handleStartClick);
    elements.changeTaskBtn.addEventListener('click', handleChangeTaskClick);
    elements.tabs.forEach(tab => tab.addEventListener('click', handleTabClick));
    elements.showHeatmap.addEventListener('change', handleHeatmapToggle);
    elements.showHeatmap.addEventListener('change', handleHeatmapToggle);
    elements.imageSearch.addEventListener('input', handleImageSearch);
    elements.btnSelectAll.addEventListener('click', handleSelectAll);
    elements.btnDeselectAll.addEventListener('click', handleDeselectAll);
    elements.btnSelectAllModels.addEventListener('click', handleSelectAllModels);
    elements.btnDeselectAllModels.addEventListener('click', handleDeselectAllModels);


    // Double-click on task to start
    elements.taskSelect.addEventListener('dblclick', handleStartClick);
}

// Start the app
init();
