// Global state
let state = {
    // Setup
    annotators: [],
    tasks: [],
    currentAnnotator: '',
    currentTask: null,     // Current annotation task config
    activeAnnotationModes: [], // Annotation modes from task config

    // Prompts and images
    prompts: [],           // All prompts from selected JSON file
    currentPromptIndex: 0, // Current prompt being evaluated
    currentPrompt: null,   // Current prompt data (with annotations)

    // Anchor configuration
    anchorConfig: null,    // Loaded from needs_anchor.json

    // Images for current prompt (across all models and aspect ratios)
    imagesForPrompt: [],   // All images for current prompt
    currentImageIndex: 0,  // Current image index within prompt
    currentImage: null,    // Current image metadata

    // Annotations
    userAnswers: {},
    chatgptAnswers: {},
    modifiedAnswers: new Set(),
    aestheticsRating: 0,

    // Artifact annotation
    isAnnotatingArtifacts: false,
    canvas: null,
    ctx: null,
    isDrawing: false,
    isEraseMode: false,

    // Artifact point annotation
    artifactPoints: [],        // List of {x, y} in image-space coordinates
    isPointRemoveMode: false,  // Toggle between add/remove

    // Batch VQA processing
    batchVqaRunning: false,
    batchVqaQueue: [],
    batchVqaCancelled: false,

    // All images list (flat list of {prompt, image} pairs - optionally shuffled)
    isShuffleMode: false,
    allImages: [],         // Flat list of {prompt, image} pairs
    allImagesIndex: 0,     // Current position in images list

    // Mask overlay visibility
    maskVisible: true      // Whether artifact mask overlay is visible
};

// ============ COOKIE HELPERS ============

// Get cookie value by name
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

// Set cookie with name, value, and expiration days
function setCookie(name, value, days = 365) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    const expires = `expires=${date.toUTCString()}`;
    document.cookie = `${name}=${value}; ${expires}; path=/`;
}

// Delete cookie by name
function deleteCookie(name) {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/`;
}

const API_BASE = '';

// ============ ANNOTATION MODE HELPERS ============

// Check if any BQA questions should be shown
function shouldShowBqaQuestions() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true; // Default: show all
    return modes.includes('all_bqa') || modes.includes('bqa_anchor') || modes.includes('bqa_no_anchor');
}

// Check if anchors should be visible (not hidden)
function shouldShowAnchors() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true; // Default: show anchors
    if (modes.includes('bqa_no_anchor')) return false;
    return modes.includes('all_bqa') || modes.includes('bqa_anchor') || modes.includes('likert_anchor');
}

// Check if likert anchor mode is enabled
function shouldShowAnchorLikert() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true;
    return modes.includes('likert_anchor');
}

// Check if text rendering should use per-word mode (vs binary)
function shouldShowTextPerWord() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true; // Default: per-word
    return modes.includes('text_per_word');
}

// Check if text rendering should use likert mode
function shouldShowTextLikert() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return false;
    return modes.includes('text_likert') || modes.includes('likert_anchor');
}

// Check if text rendering should use binary BQA mode
function shouldShowTextBqa() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return false; // Default: per-word, not binary
    return modes.includes('text_bqa');
}

// Check if any text rendering questions should be shown
function shouldShowTextRendering() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true; // Default: show all
    return modes.includes('text_bqa') || modes.includes('text_per_word') || modes.includes('text_likert') || modes.includes('likert_anchor');
}

// Get generic likert question from annotation mode string
function getGenericLikertQuestion() {
    const modes = state.activeAnnotationModes;
    if (!modes) return null;

    // Look for string starting with "likert_question@"
    const genericMode = modes.find(m => m.startsWith('likert_question@'));
    if (genericMode) {
        return genericMode.split('likert_question@')[1];
    }
    return null;
}

// Check if aesthetics rating should be shown
function shouldShowAesthetics() {
    const modes = state.activeAnnotationModes;
    if (!modes || modes.length === 0) return true; // Default: show all

    // Show if artifacts_likert is present OR a generic likert question is defined
    return modes.includes('artifacts_likert') || getGenericLikertQuestion() !== null;
}


// Check if artifact annotation should be shown (only for artifacts_brush mode)
function shouldShowArtifacts() {
    const modes = state.activeAnnotationModes;

    // Explicitly enable for artifacts_brush mode (overrides other disable logic)
    if (modes && modes.includes('artifacts_brush')) return true;

    // Default: show all if no modes specified
    if (!modes || modes.length === 0) return true;

    // Otherwise disable (including artifacts_likert mode or generic likert mode)
    return false;
}

// Check if artifact point annotation should be shown
function shouldShowArtifactsPoint() {
    const modes = state.activeAnnotationModes;
    if (modes && modes.includes('artifacts_point')) return true;
    return false;
}

// Check if BQA AI is enabled for the current task
function isBqaAiEnabled() {
    if (!state.currentTask) return true; // Default: enabled
    // Check for explicit enable_bqa_ai setting (default to true if not set)
    return state.currentTask.enable_bqa_ai !== false;
}

// Check if a specific annotation should be shown based on modes
function shouldShowAnnotation(annotation) {
    const modes = state.activeAnnotationModes;

    // Default: show all if no modes specified
    if (!modes || modes.length === 0) return true;

    const isTextRendering = annotation.skill === 'text_rendering' &&
        (annotation.subskill === 'rendering_accuracy' || annotation.subskill === 'numerical');

    // Check if annotation has an anchor
    const hasAnchor = checkIfHasAnchor(annotation);

    // Text rendering questions
    if (isTextRendering) {
        if (modes.includes('likert_anchor') && hasAnchor) return true;
        return modes.includes('text_bqa') || modes.includes('text_per_word') || modes.includes('text_likert');
    }

    if (modes.includes('all_bqa')) return true;
    if (modes.includes('bqa_anchor') && hasAnchor) return true;
    if (modes.includes('likert_anchor') && hasAnchor) return true;
    if (modes.includes('bqa_no_anchor') && hasAnchor) return true;

    return false;
}

// Check if annotation has anchor (without needing anchor config to be loaded yet)
function checkIfHasAnchor(annotation) {
    if (!state.anchorConfig || !state.anchorConfig.annotations) return false;
    if (!state.currentPrompt) return false;

    const promptId = state.currentPrompt.prompt_id;
    const uid = annotation.uid;

    const configEntry = state.anchorConfig.annotations.find(
        entry => entry.prompt_id === promptId && entry.uid === uid
    );

    return configEntry && configEntry.needs_anchor === true;
}

// Check if annotation has anchor for a specific prompt (used for filtering)
function checkIfHasAnchorForPrompt(annotation, promptId) {
    if (!state.anchorConfig || !state.anchorConfig.annotations) return false;

    const uid = annotation.uid;

    const configEntry = state.anchorConfig.annotations.find(
        entry => entry.prompt_id === promptId && entry.uid === uid
    );

    return configEntry && configEntry.needs_anchor === true;
}

// Check if a prompt has any relevant annotations for the current mode
function promptHasRelevantAnnotations(prompt) {
    const modes = state.activeAnnotationModes;

    // If no modes specified (default mode), always show
    if (!modes || modes.length === 0) return true;

    // For artifacts-only modes, always show (aesthetics rating always available)
    if (modes.includes('artifacts_likert') || modes.includes('artifacts_brush') || modes.includes('artifacts_point') || getGenericLikertQuestion() !== null) {
        return true;
    }

    // Check if any annotations in this prompt match the modes
    const annotations = prompt.annotations || [];
    const promptId = prompt.prompt_id;

    for (const annotation of annotations) {
        const isTextRendering = annotation.skill === 'text_rendering' &&
            (annotation.subskill === 'rendering_accuracy' || annotation.subskill === 'numerical');

        // Check if annotation has an anchor
        const hasAnchor = checkIfHasAnchorForPrompt(annotation, promptId);

        // Text rendering questions
        if (isTextRendering) {
            if (modes.includes('text_bqa') || modes.includes('text_per_word') || modes.includes('text_likert')) return true;
            continue;
        }

        // Non-text rendering (BQA) questions
        if (modes.includes('all_bqa')) return true;
        if ((modes.includes('bqa_anchor') || modes.includes('bqa_no_anchor') || modes.includes('likert_anchor')) && hasAnchor) return true;
    }

    return false;
}

// Render Text Likert UI
function renderTextLikertUI(annotation, questionDiv, isDisabled) {
    const uid = annotation.uid;
    const container = document.createElement('div');
    container.className = 'aesthetics-rating-section'; // Reuse existing style class
    container.style.marginTop = '8px';
    container.style.border = 'none'; // Remove border for cleaner look inside question
    container.style.background = 'transparent';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '10px';

    const starRating = document.createElement('div');
    starRating.className = 'star-rating';
    starRating.id = `text-likert-${uid}`;

    // Get existing value
    const savedValue = state.userAnswers[uid] !== undefined ? state.userAnswers[uid] : null;

    // Create "No Text" button
    const noTextBtn = document.createElement('button');
    noTextBtn.className = 'no-text-btn';
    noTextBtn.textContent = 'No Text';
    noTextBtn.title = 'Rate as 0 (No Text)';

    // Check if 0 is selected
    if (savedValue === 0) {
        noTextBtn.classList.add('selected');
    }

    if (isDisabled) {
        noTextBtn.disabled = true;
        noTextBtn.style.cursor = 'default';
    } else {
        noTextBtn.addEventListener('click', () => {
            // Set value to 0
            state.userAnswers[uid] = 0;
            state.modifiedAnswers.add(uid);

            // Update UI
            noTextBtn.classList.add('selected');
            const stars = starRating.querySelectorAll('.star');
            stars.forEach(s => s.classList.remove('selected'));

            // Update dependent questions when answer changes
            if (state.currentPrompt && state.currentPrompt.annotations) {
                updateDependentQuestions(uid, state.currentPrompt.annotations);
            }
        });
    }

    // Create 5 stars
    [1, 2, 3, 4, 5].forEach(rating => {
        const star = document.createElement('span');
        star.className = 'star';
        star.textContent = '★';
        star.dataset.rating = rating;

        if (savedValue !== null && rating <= savedValue) {
            star.classList.add('selected');
        }

        if (!isDisabled) {
            star.addEventListener('click', () => {
                // Update specific question answer
                state.userAnswers[uid] = rating;
                state.modifiedAnswers.add(uid);

                // Update UI
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => {
                    const r = parseInt(s.dataset.rating);
                    if (r <= rating) {
                        s.classList.add('selected');
                    } else {
                        s.classList.remove('selected');
                    }
                });
                // Deselect No Text button
                noTextBtn.classList.remove('selected');

                // Update dependent questions when answer changes
                if (state.currentPrompt && state.currentPrompt.annotations) {
                    updateDependentQuestions(uid, state.currentPrompt.annotations);
                }
            });

            star.addEventListener('mouseenter', () => {
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => {
                    if (parseInt(s.dataset.rating) <= rating) {
                        s.classList.add('hover');
                    } else {
                        s.classList.remove('hover');
                    }
                });
            });

            star.addEventListener('mouseleave', () => {
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => s.classList.remove('hover'));
            });
        } else {
            star.style.cursor = 'default';
        }

        starRating.appendChild(star);
    });

    container.appendChild(noTextBtn);
    container.appendChild(starRating);
    questionDiv.appendChild(container);
    return true;
}

// Render Anchor Likert UI (1-5 scale + Wrong button)
function renderAnchorLikertUI(annotation, questionDiv, isDisabled) {
    const uid = annotation.uid;
    const container = document.createElement('div');
    container.className = 'aesthetics-rating-section'; // Reuse existing style class
    container.style.marginTop = '8px';
    container.style.border = 'none'; // Remove border for cleaner look inside question
    container.style.background = 'transparent';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.gap = '10px';

    const starRating = document.createElement('div');
    starRating.className = 'star-rating';
    starRating.id = `anchor-likert-${uid}`;

    // Add anchor icon if needed (to the left of Wrong button)
    if (shouldShowAnchors() && shouldShowAnchorImage(annotation)) {
        const anchorIndicator = document.createElement('span');
        anchorIndicator.className = 'anchor-indicator';
        anchorIndicator.title = 'Hover to see reference image';
        anchorIndicator.dataset.uid = uid;
        anchorIndicator.addEventListener('mouseenter', (e) => showAnchorImage(annotation, e.target));
        anchorIndicator.addEventListener('mouseleave', hideAnchorImage);

        container.appendChild(anchorIndicator);
    }

    // Get existing value
    const savedValue = state.userAnswers[uid] !== undefined ? state.userAnswers[uid] : null;

    // Create "Wrong" button (behaves like "No Text", saves as 0)
    const wrongBtn = document.createElement('button');
    wrongBtn.className = 'no-text-btn';
    wrongBtn.textContent = 'Wrong';
    wrongBtn.title = 'Rate as 0 (Wrong)';

    // Check if 0 is selected
    if (savedValue === 0) {
        wrongBtn.classList.add('selected');
    }

    // Create "Unsure" button
    const unsureBtn = document.createElement('button');
    unsureBtn.className = 'no-text-btn'; // Reuse same style
    unsureBtn.textContent = 'Unsure';
    unsureBtn.title = 'Rate as Unsure';
    unsureBtn.style.marginLeft = '5px';

    // Check if "unsure" is selected
    if (savedValue === 'unsure') {
        unsureBtn.classList.add('selected');
    }

    if (isDisabled) {
        wrongBtn.disabled = true;
        wrongBtn.style.cursor = 'default';
        unsureBtn.disabled = true;
        unsureBtn.style.cursor = 'default';
    } else {
        wrongBtn.addEventListener('click', () => {
            // Set value to 0
            state.userAnswers[uid] = 0;
            state.modifiedAnswers.add(uid);

            // Update UI
            wrongBtn.classList.add('selected');
            unsureBtn.classList.remove('selected');
            const stars = starRating.querySelectorAll('.star');
            stars.forEach(s => s.classList.remove('selected'));

            // Update dependent questions when answer changes
            if (state.currentPrompt && state.currentPrompt.annotations) {
                updateDependentQuestions(uid, state.currentPrompt.annotations);
            }
        });

        unsureBtn.addEventListener('click', () => {
            // Set value to "unsure"
            state.userAnswers[uid] = 'unsure';
            state.modifiedAnswers.add(uid);

            // Update UI
            unsureBtn.classList.add('selected');
            wrongBtn.classList.remove('selected');
            const stars = starRating.querySelectorAll('.star');
            stars.forEach(s => s.classList.remove('selected'));

            // Update dependent questions when answer changes
            if (state.currentPrompt && state.currentPrompt.annotations) {
                updateDependentQuestions(uid, state.currentPrompt.annotations);
            }
        });
    }

    // Create 5 stars
    [1, 2, 3, 4, 5].forEach(rating => {
        const star = document.createElement('span');
        star.className = 'star';
        star.textContent = '★';
        star.dataset.rating = rating;

        if (savedValue !== null && typeof savedValue === 'number' && rating <= savedValue) {
            star.classList.add('selected');
        }

        if (!isDisabled) {
            star.addEventListener('click', () => {
                // Update specific question answer
                state.userAnswers[uid] = rating;
                state.modifiedAnswers.add(uid);

                // Update UI
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => {
                    const r = parseInt(s.dataset.rating);
                    if (r <= rating) {
                        s.classList.add('selected');
                    } else {
                        s.classList.remove('selected');
                    }
                });
                // Deselect Wrong and Unsure buttons
                wrongBtn.classList.remove('selected');
                unsureBtn.classList.remove('selected');

                // Update dependent questions when answer changes
                if (state.currentPrompt && state.currentPrompt.annotations) {
                    updateDependentQuestions(uid, state.currentPrompt.annotations);
                }
            });

            star.addEventListener('mouseenter', () => {
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => {
                    if (parseInt(s.dataset.rating) <= rating) {
                        s.classList.add('hover');
                    } else {
                        s.classList.remove('hover');
                    }
                });
            });

            star.addEventListener('mouseleave', () => {
                const stars = starRating.querySelectorAll('.star');
                stars.forEach(s => s.classList.remove('hover'));
            });
        } else {
            star.style.cursor = 'default';
        }

        starRating.appendChild(star);
    });

    container.appendChild(wrongBtn);
    container.appendChild(starRating);
    container.appendChild(unsureBtn);
    questionDiv.appendChild(container);
    return true;
}


// Update UI visibility based on annotation modes
function updateUIVisibility() {
    const modes = state.activeAnnotationModes;
    const showAll = !modes || modes.length === 0;

    // Aesthetics section
    const aestheticsSection = document.querySelector('.aesthetics-rating-section');
    if (aestheticsSection) {
        aestheticsSection.style.display = (showAll || shouldShowAesthetics()) ? 'block' : 'none';

        // Update label based on generic likert question if present
        const genericQuestion = getGenericLikertQuestion();
        const labelEl = document.getElementById('aesthetics-question-label');
        if (labelEl) {
            labelEl.textContent = genericQuestion ? genericQuestion : 'How do you like the aesthetics of the image?';
        }
    }

    // Artifact brush button
    const artifactButton = document.getElementById('artifact-brush-button');
    if (artifactButton) {
        artifactButton.style.display = (showAll || shouldShowArtifacts()) ? 'flex' : 'none';
    }

    // Artifact point button
    const artifactPointButton = document.getElementById('artifact-point-button');
    if (artifactPointButton) {
        artifactPointButton.style.display = shouldShowArtifactsPoint() ? 'flex' : 'none';
    }

    // Mask toggle button
    const maskToggleButton = document.getElementById('mask-toggle-button');
    if (maskToggleButton) {
        maskToggleButton.style.display = (showAll || shouldShowArtifacts() || shouldShowArtifactsPoint()) ? 'flex' : 'none';
    }

    // Batch VQA button - hide if AI is disabled
    const batchVqaButton = document.getElementById('btn-batch-vqa');
    if (batchVqaButton) {
        batchVqaButton.style.display = isBqaAiEnabled() ? 'inline-flex' : 'none';
    }

    // Status checkboxes - update visibility
    const vqaCheckbox = document.getElementById('status-vqa');
    const artifactsCheckbox = document.getElementById('status-artifacts');
    const aestheticsCheckbox = document.getElementById('status-aesthetics');

    if (vqaCheckbox && vqaCheckbox.parentElement) {
        const showVqa = showAll || shouldShowBqaQuestions() || shouldShowTextRendering() || shouldShowAnchorLikert();
        vqaCheckbox.parentElement.style.display = showVqa ? 'flex' : 'none';
    }
    if (artifactsCheckbox && artifactsCheckbox.parentElement) {
        artifactsCheckbox.parentElement.style.display = (showAll || shouldShowArtifacts() || shouldShowArtifactsPoint()) ? 'flex' : 'none';
    }
    if (aestheticsCheckbox && aestheticsCheckbox.parentElement) {
        aestheticsCheckbox.parentElement.style.display = (showAll || shouldShowAesthetics()) ? 'flex' : 'none';
    }

    // Questions container visibility (handled in displayQuestions)
    const questionsContainer = document.getElementById('questions-container');
    if (questionsContainer) {
        const showQuestions = showAll || shouldShowBqaQuestions() || shouldShowTextRendering() || shouldShowAnchorLikert();
        questionsContainer.style.display = showQuestions ? 'block' : 'none';
    }
}

// ============ INITIALIZATION ============

async function init() {
    await loadSetupData();
    setupEventListeners();
    showSetupModal();
}

async function loadSetupData() {
    try {
        // Load annotators
        const annotatorsRes = await fetch(`${API_BASE}/api/annotators`);
        const annotatorsData = await annotatorsRes.json();
        state.annotators = annotatorsData.annotators;

        const annotatorSelect = document.getElementById('annotator-select');

        // Get saved annotator from cookie
        const savedAnnotator = getCookie('image_eval_annotator');

        state.annotators.forEach(annotator => {
            const option = document.createElement('option');
            option.value = annotator;
            option.textContent = annotator;
            // Pre-select saved annotator
            if (savedAnnotator && annotator === savedAnnotator) {
                option.selected = true;
            }
            annotatorSelect.appendChild(option);
        });

        // Load annotation tasks
        const tasksRes = await fetch(`${API_BASE}/api/annotation_tasks`);
        const tasksData = await tasksRes.json();
        state.tasks = tasksData.tasks;

        const taskSelect = document.getElementById('task-select');
        state.tasks.forEach(task => {
            const option = document.createElement('option');
            option.value = task.id;
            option.textContent = task.name;
            option.dataset.taskId = task.id;
            taskSelect.appendChild(option);
        });

        // Show details for first task
        if (state.tasks.length > 0) {
            showTaskDetails(state.tasks[0]);
        }

        // Update task completion checkmarks if we have a saved annotator
        if (savedAnnotator && state.annotators.includes(savedAnnotator)) {
            await updateTaskCompletionCheckmarks(savedAnnotator);
        }

        // Add listener to annotator select to update checkmarks when changed
        annotatorSelect.addEventListener('change', async () => {
            const selectedAnnotator = annotatorSelect.value;
            if (selectedAnnotator) {
                await updateTaskCompletionCheckmarks(selectedAnnotator);
            }
        });
    } catch (error) {
        console.error('Error loading setup data:', error);
    }
}

// Update task completion checkmarks in the dropdown
async function updateTaskCompletionCheckmarks(annotator) {
    const taskSelect = document.getElementById('task-select');
    const options = taskSelect.querySelectorAll('option');

    for (const option of options) {
        const taskId = option.value;
        if (!taskId) continue;

        try {
            const response = await fetch(
                `${API_BASE}/api/task/${taskId}/completion_status?annotator=${encodeURIComponent(annotator)}`
            );
            if (response.ok) {
                const data = await response.json();
                const task = state.tasks.find(t => t.id === taskId);
                const taskName = task ? task.name : taskId;

                if (data.is_complete) {
                    option.textContent = `${taskName} ✓`;
                    option.classList.add('task-complete');
                } else {
                    option.textContent = taskName;
                    option.classList.remove('task-complete');
                }
            }
        } catch (error) {
            console.error(`Error checking completion for task ${taskId}:`, error);
        }
    }
}

function showTaskDetails(task) {
    const detailsDiv = document.getElementById('task-details');
    detailsDiv.innerHTML = `
        <div class="task-info">
            <div><strong>Dataset:</strong> ${task.dataset_version}</div>
            <div><strong>Prompts:</strong> ${task.prompts_file}</div>
            <div><strong>Models:</strong> ${task.models.join(', ')}</div>
        </div>
    `;
}

function showSetupModal() {
    document.getElementById('setup-modal').style.display = 'flex';
}

async function startAnnotation() {
    const annotator = document.getElementById('annotator-select').value;
    const taskId = document.getElementById('task-select').value;

    if (!annotator) {
        alert('Please select an annotator');
        return;
    }
    if (!taskId) {
        alert('Please select an annotation task');
        return;
    }

    state.currentAnnotator = annotator;
    state.currentTask = state.tasks.find(t => t.id === taskId);

    // Save selected annotator to cookie for next time
    setCookie('image_eval_annotator', annotator);

    // Set annotation modes from task config (default to empty = show all)
    state.activeAnnotationModes = state.currentTask.annotations || [];

    // Check if shuffle mode is enabled
    state.isShuffleMode = state.currentTask.shuffle_images === true;

    // Update header
    document.getElementById('annotator-name').textContent = annotator;

    // Display task name in header
    const taskNameEl = document.getElementById('task-name-display');
    if (taskNameEl) {
        taskNameEl.textContent = state.currentTask.name;
    }

    // Update UI visibility based on annotation modes
    updateUIVisibility();

    // Hide modal
    document.getElementById('setup-modal').style.display = 'none';

    // Load anchor configuration and prompts
    await loadAnchorConfig();
    await loadPrompts();

    // Always build flat images list (shuffled or sequential based on isShuffleMode)
    await buildAllImagesList();
    displayCurrentListImage();
}

// ============ PROMPTS LOADING ============

async function loadAnchorConfig() {
    try {
        const response = await fetch(
            `${API_BASE}/api/task/${state.currentTask.id}/anchor_config`
        );
        if (response.ok) {
            state.anchorConfig = await response.json();
            console.log('Anchor config loaded:', state.anchorConfig);
        } else {
            console.log('No anchor config found for this task');
            state.anchorConfig = null;
        }
    } catch (error) {
        console.log('Error loading anchor config:', error);
        state.anchorConfig = null;
    }
}

async function loadPrompts() {
    try {
        const response = await fetch(
            `${API_BASE}/api/task/${state.currentTask.id}/prompts`
        );
        const data = await response.json();
        state.prompts = data.prompts;

        // Find first prompt with relevant annotations for the current mode
        state.currentPromptIndex = 0;
        for (let i = 0; i < state.prompts.length; i++) {
            if (promptHasRelevantAnnotations(state.prompts[i])) {
                state.currentPromptIndex = i;
                break;
            }
        }

        if (state.prompts.length > 0) {
            state.currentPrompt = state.prompts[state.currentPromptIndex];
        }

        updateProgress();
    } catch (error) {
        console.error('Error loading prompts:', error);
    }
}

// ============ IMAGES LOADING ============

async function loadCurrentPromptImages() {
    if (!state.currentPrompt || !state.currentTask) {
        state.imagesForPrompt = [];
        return;
    }

    const promptId = state.currentPrompt.prompt_id;

    try {
        const response = await fetch(
            `${API_BASE}/api/task/${state.currentTask.id}/images_for_prompt/${promptId}?annotator=${state.currentAnnotator}`
        );
        const data = await response.json();
        state.imagesForPrompt = data.images;

        // Shuffle images if enabled in task config (default: false)
        if (state.currentTask.shuffle_images === true) {
            shuffleArray(state.imagesForPrompt);
        }

        state.currentImageIndex = 0;

        if (state.imagesForPrompt.length > 0) {
            state.currentImage = state.imagesForPrompt[0];
        } else {
            state.currentImage = null;
        }
    } catch (error) {
        console.error('Error loading images for prompt:', error);
        state.imagesForPrompt = [];
    }
}

// Fisher-Yates shuffle algorithm
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Build list of all images across all prompts (optionally shuffled)
async function buildAllImagesList() {
    state.allImages = [];

    // Get all prompts with relevant annotations
    const relevantPrompts = state.prompts.filter(p => promptHasRelevantAnnotations(p));

    // Fetch images for each prompt and build flat list
    for (const prompt of relevantPrompts) {
        try {
            const response = await fetch(
                `${API_BASE}/api/task/${state.currentTask.id}/images_for_prompt/${prompt.prompt_id}?annotator=${state.currentAnnotator}`
            );
            const data = await response.json();
            const images = data.images || [];

            // Add each image with its prompt reference
            for (const image of images) {
                state.allImages.push({
                    prompt: prompt,
                    image: image
                });
            }
        } catch (error) {
            console.error(`Error loading images for prompt ${prompt.prompt_id}:`, error);
        }
    }

    // Shuffle first if shuffle mode is enabled
    if (state.isShuffleMode) {
        shuffleArray(state.allImages);
        console.log(`Shuffle mode: Shuffled ${state.allImages.length} images from ${relevantPrompts.length} prompts`);
    }

    // Then sort images: incomplete (not annotated) first, then completed
    // This maintains randomization within each group (incomplete and complete)
    state.allImages.sort((a, b) => {
        // false (not annotated) should come before true (annotated)
        // This converts false to 0 and true to 1, so incomplete images sort first
        return (a.image.annotated ? 1 : 0) - (b.image.annotated ? 1 : 0);
    });

    if (!state.isShuffleMode) {
        console.log(`Built ${state.allImages.length} images from ${relevantPrompts.length} prompts (sequential order, incomplete first)`);
    }

    state.allImagesIndex = 0;

    // Pre-populate completion cache for all images (one-time cost at startup)
    // This avoids per-image server fetches during updateListProgress
    for (const item of state.allImages) {
        item.image._completionCached = await checkSingleImageCompletion(item.image);
    }
    console.log(`Completion cache populated for ${state.allImages.length} images`);
}

// Display current image from the all-images list
async function displayCurrentListImage() {
    if (state.allImages.length === 0 || state.allImagesIndex >= state.allImages.length) {
        document.getElementById('main-image').src = '';
        document.getElementById('questions-container').innerHTML =
            '<p class="no-images">No images found.</p>';
        return;
    }

    const current = state.allImages[state.allImagesIndex];
    state.currentPrompt = current.prompt;
    state.currentImage = current.image;

    // Also update imagesForPrompt for compatibility (single image)
    state.imagesForPrompt = [current.image];
    state.currentImageIndex = 0;

    // Update header to show prompt ID only (no model indicator to avoid bias)
    document.getElementById('current-prompt-id').textContent = current.prompt.prompt_id;

    // Display prompt text
    document.getElementById('prompt-text').textContent = current.prompt.prompt || 'No prompt loaded';

    // Display the image
    const imgElement = document.getElementById('main-image');
    imgElement.src = `${API_BASE}/generated_images/${current.image.image_path}`;

    imgElement.onload = () => {
        setupCanvas();
    };

    // Load annotations for this specific image
    await loadImageAnnotations();

    // Update status checkboxes
    document.getElementById('status-vqa').checked = current.image.annotated;

    // Update progress
    await updateListProgress();
}

// Update progress display for the all-images list (uses cached completion status)
async function updateListProgress() {
    const total = state.allImages.length;
    const current = state.allImagesIndex + 1;

    // Count completed images using locally cached completion flags
    // (populated during buildAllImagesList and updated on save)
    let completed = 0;
    for (const item of state.allImages) {
        if (item.image._completionCached) {
            completed++;
        }
    }

    const percentage = total > 0 ? (completed / total) * 100 : 0;
    document.getElementById('header-progress-fill').style.width = `${percentage}%`;
    document.getElementById('header-progress-text').textContent = `${current} / ${total}`;

    // Update back button
    const backBtn = document.getElementById('btn-back');
    backBtn.disabled = state.allImagesIndex <= 0;
}

// ============ DISPLAY ============

async function displayCurrentImage() {
    // Update header info
    document.getElementById('current-prompt-id').textContent =
        state.currentPrompt ? state.currentPrompt.prompt_id : '-';

    // Display prompt text
    const promptText = state.currentPrompt ? state.currentPrompt.prompt : 'No prompt loaded';
    document.getElementById('prompt-text').textContent = promptText;

    if (!state.currentImage) {
        document.getElementById('main-image').src = '';
        document.getElementById('questions-container').innerHTML =
            '<p class="no-images">No images found for this prompt.</p>';
        updateProgress();
        return;
    }

    // Display image
    const imgElement = document.getElementById('main-image');
    imgElement.src = `${API_BASE}/generated_images/${state.currentImage.image_path}`;

    // Setup canvas on image load
    imgElement.onload = () => {
        setupCanvas();
    };

    // Load annotations for this specific image (this will also call displayQuestions when done)
    await loadImageAnnotations();

    // Update status checkboxes
    document.getElementById('status-vqa').checked = state.currentImage.annotated;

    updateProgress();
}

async function loadImageAnnotations() {
    // Reset mask visibility to "On" when loading a new image
    state.maskVisible = true;

    if (!state.currentImage || !state.currentTask) return;

    try {
        const response = await fetch(
            `${API_BASE}/api/image/${state.currentTask.id}/${state.currentImage.model}/${state.currentImage.prompt_id}?annotator=${state.currentAnnotator}&t=${Date.now()}`
        );
        const data = await response.json();

        state.userAnswers = data.user_annotations || {};

        // Map _text_likert back to UID if present
        if (state.userAnswers && state.userAnswers['_text_likert'] !== undefined && state.currentPrompt) {
            const annotations = state.currentPrompt.annotations || [];
            for (const annot of annotations) {
                const isTextRendering = annot.skill === 'text_rendering' &&
                    (annot.subskill === 'rendering_accuracy' || annot.subskill === 'numerical');
                if (isTextRendering) {
                    state.userAnswers[annot.uid] = state.userAnswers['_text_likert'];
                }
            }
        }

        // Only load chatgpt answers if AI is enabled for this task
        state.chatgptAnswers = isBqaAiEnabled() ? (data.chatgpt_annotations || {}) : {};
        state.modifiedAnswers.clear();

        // Load aesthetics rating (stored in user_annotations)
        state.aestheticsRating = (data.user_annotations && data.user_annotations._aesthetics_rating) || 0;
        updateStarDisplay(state.aestheticsRating);
        document.getElementById('status-aesthetics').checked = state.aestheticsRating > 0;

        // Update artifact checkbox
        const hasAnyArtifactAnnotation = data.has_artifact_mask || data.has_artifact_points;
        document.getElementById('status-artifacts').checked = hasAnyArtifactAnnotation;

        // Show or hide artifact mask overlay on main image
        const maskOverlay = document.getElementById('artifact-mask-overlay');

        // Add error handler if not present (to hide broken image icon)
        maskOverlay.onerror = function () {
            this.style.display = 'none';
        };

        if (data.has_artifact_mask && state.currentTask && state.currentImage) {
            // Build mask path: /annotations/<task_id>/artifact_mask/<model>/<prompt_id>_<annotator>.png
            const maskPath = `${API_BASE}/annotations/${state.currentTask.id}/artifact_mask/${state.currentImage.model}/${state.currentImage.prompt_id}_${state.currentAnnotator}.png`;
            maskOverlay.src = maskPath;
        } else {
            maskOverlay.removeAttribute('src');
            maskOverlay.style.display = 'none';
        }
        // Update visibility based on toggle state
        updateMaskVisibility();

        // Load artifact points and render overlay
        // Backend returns tuples [[x,y],...], convert to {x,y} objects for rendering
        const rawPoints = data.artifact_points || [];
        state.artifactPoints = rawPoints.map(p => Array.isArray(p) ? { x: p[0], y: p[1] } : p);
        renderPointOverlay();

        // If no chatgpt annotations and AI is enabled, generate them
        if (!data.has_chatgpt && state.currentPrompt && isBqaAiEnabled()) {
            await generateChatGPTAnnotations();
        } else {
            // Display questions with loaded annotations
            displayQuestions();
        }
    } catch (error) {
        console.error('Error loading image annotations:', error);
        state.userAnswers = {};
        state.chatgptAnswers = {};
        displayQuestions();
    }
}

async function generateChatGPTAnnotations() {
    if (!state.currentImage || !state.currentPrompt) return;

    // Don't generate if AI is disabled
    if (!isBqaAiEnabled()) return;

    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch(
            `${API_BASE}/api/generate_chatgpt_annotations/${state.currentTask.id}/${state.currentImage.model}/${state.currentImage.prompt_id}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt_data: state.currentPrompt,
                    dataset_version: state.currentTask.dataset_version,
                    preprocess_with_al: false
                })
            }
        );
        const data = await response.json();

        if (data.success) {
            state.chatgptAnswers = data.annotations;
            displayQuestions(); // Refresh questions with new answers
        }
    } catch (error) {
        console.error('Error generating ChatGPT annotations:', error);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Check if a question should be disabled based on its dependencies
function isQuestionDisabled(annotation, annotations) {
    const dependsOn = annotation.depends_on || [];
    if (dependsOn.length === 0) return false;

    // Check each dependency - if any parent answer is 'no', disable this question
    for (const parentUid of dependsOn) {
        // Find the parent annotation
        const parentAnnotation = annotations.find(a => a.uid === parentUid);

        // If parent is hidden (not in current annotation mode), assume dependency is satisfied
        // This prevents child questions from being disabled due to hidden parent questions
        if (parentAnnotation && !shouldShowAnnotation(parentAnnotation)) {
            continue; // Skip this dependency check
        }

        let parentAnswer = state.userAnswers[parentUid];
        if (parentAnswer === undefined) {
            parentAnswer = state.chatgptAnswers[parentUid];
        }

        const isCountExact = parentAnnotation && parentAnnotation.skill === 'entities' && parentAnnotation.subskill === 'count_exact';

        if (isCountExact) {
            // Relax dependency for count_exact: 'no' answer (wrong count) shouldn't disable child questions
            if (parentAnswer === 0 || parentAnswer === 'unsure') {
                return true;
            }
        } else {
            if (parentAnswer === 'no' || parentAnswer === 0 || parentAnswer === 'unsure') {
                return true;
            }
        }
    }
    return false;
}

// Get the phrase of a parent annotation by uid
function getParentPhrase(parentUid, annotations) {
    const parent = annotations.find(a => a.uid === parentUid);
    return parent ? parent.phrase : parentUid;
}

// Update disabled state of all dependent questions
function updateDependentQuestions(changedUid, annotations) {
    annotations.forEach(annotation => {
        const dependsOn = annotation.depends_on || [];
        if (dependsOn.includes(changedUid)) {
            const questionDiv = document.getElementById(`question-item-${annotation.uid}`);
            const checkbox = document.getElementById(`toggle-${annotation.uid}`);
            if (questionDiv && checkbox) {
                const isDisabled = isQuestionDisabled(annotation, annotations);
                questionDiv.classList.toggle('question-disabled', isDisabled);
                checkbox.disabled = isDisabled;
            }
            // Recursively update questions that depend on this one
            updateDependentQuestions(annotation.uid, annotations);
        }
    });
}

// ============ PER-WORD TEXT RENDERING EVALUATION ============

// Check if annotation should use per-word text rendering evaluation
function isPerWordTextRendering(annotation) {
    return annotation.skill === 'text_rendering' &&
        (annotation.subskill === 'rendering_accuracy' || annotation.subskill === 'numerical');
}

// Extract words from question text (text between single or double quotes)
function extractWordsFromQuestion(question) {
    if (!question) return [];

    // Prioritize double quotes (more reliable for text with apostrophes)
    // Match text between double quotes first
    const doubleQuoteMatch = question.match(/"([^"]+)"/);
    if (doubleQuoteMatch && doubleQuoteMatch[1]) {
        const quotedText = doubleQuoteMatch[1];
        return quotedText.split(/\s+/).filter(word => word.length > 0);
    }

    // Fall back to single quotes only if no double quotes found
    // Use a more specific pattern that expects surrounding context
    const singleQuoteMatch = question.match(/\btext\s+'([^']+)'/i) ||
        question.match(/\bcontain\s+'([^']+)'/i) ||
        question.match(/'([^']+)'\s*\?/);
    if (singleQuoteMatch && singleQuoteMatch[1]) {
        const quotedText = singleQuoteMatch[1];
        return quotedText.split(/\s+/).filter(word => word.length > 0);
    }

    return [];
}

// Get per-word answer for a specific question
function getPerWordAnswer(uid) {
    const answer = state.userAnswers[uid];
    // Per-word answers are stored as objects with word keys and "yes"/"no" values
    if (answer && typeof answer === 'object' && !Array.isArray(answer)) {
        return answer;
    }
    return null;
}

// Set per-word answer for a specific word
function setPerWordAnswer(uid, word, isCorrect) {
    if (!state.userAnswers[uid] || typeof state.userAnswers[uid] !== 'object' || Array.isArray(state.userAnswers[uid])) {
        // Initialize if not exists or wrong format
        state.userAnswers[uid] = {};
    }
    // Store as "yes" or "no" instead of boolean
    state.userAnswers[uid][word] = isCorrect ? 'yes' : 'no';
    // Mark as modified since we don't have AI answers for these
    state.modifiedAnswers.add(uid);
}

// Set space answer for a specific index
function setSpaceAnswer(uid, index, isClean) {
    if (!state.userAnswers[uid] || typeof state.userAnswers[uid] !== 'object' || Array.isArray(state.userAnswers[uid])) {
        state.userAnswers[uid] = {};
    }

    if (!state.userAnswers[uid]._text_artifacts_) {
        state.userAnswers[uid]._text_artifacts_ = {};
    }

    // Store as "no" (clean) or "yes" (unwanted words/artifact)
    // INVERTED LOGIC: 'no' = clean, 'yes' = artifact
    state.userAnswers[uid]._text_artifacts_[index.toString()] = isClean ? 'no' : 'yes';
    state.modifiedAnswers.add(uid);
}

// Render per-word UI for text rendering questions
function renderPerWordUI(annotation, questionDiv, isDisabled) {
    const uid = annotation.uid;
    const words = extractWordsFromQuestion(annotation.question);

    if (words.length === 0) {
        // Fallback to regular toggle if no words found
        return false;
    }

    // Create per-word container
    const perWordContainer = document.createElement('div');
    perWordContainer.className = 'per-word-container';

    // Get existing per-word answer or initialize all as correct ("yes")
    let perWordAnswer = getPerWordAnswer(uid);
    if (!perWordAnswer) {
        perWordAnswer = {};
        words.forEach((word, index) => {
            perWordAnswer[`${index}_${word}`] = 'yes';
        });
        state.userAnswers[uid] = perWordAnswer;
    }

    // Check if all words are marked as "no" (incorrect)
    const allWordsNo = words.every((word, index) => {
        const key = `${index}_${word}`;
        return perWordAnswer[key] === 'no';
    });



    // Create "No Text" button
    const noTextBtn = document.createElement('button');
    noTextBtn.className = 'no-text-btn';
    noTextBtn.textContent = 'No Text';
    noTextBtn.title = 'Mark all words as missing';
    if (allWordsNo) {
        noTextBtn.classList.add('selected');
    }

    if (isDisabled) {
        noTextBtn.disabled = true;
        noTextBtn.style.cursor = 'default';
    } else {
        noTextBtn.addEventListener('click', () => {
            // Set all words to "no"
            words.forEach((word, index) => {
                const key = `${index}_${word}`;
                setPerWordAnswer(uid, key, false); // false = "no"

                // Update specific word box UI
                const wordBox = perWordContainer.querySelector(`.word-box[data-key="${key}"]`);
                if (wordBox) {
                    wordBox.classList.remove('correct');
                    wordBox.classList.add('incorrect');
                }
            });

            // Update button state
            noTextBtn.classList.add('selected');
        });
    }

    questionDiv.appendChild(perWordContainer);
    questionDiv.appendChild(noTextBtn);

    // Ensure space answers are initialized
    if (!state.userAnswers[uid]._text_artifacts_) {
        state.userAnswers[uid]._text_artifacts_ = {};
        // Initialize N+1 spaces (before each word + after last)
        for (let i = 0; i <= words.length; i++) {
            state.userAnswers[uid]._text_artifacts_[i.toString()] = 'no'; // Default to clean (NO artifact)
        }
    }

    // Helper to create space box
    const createSpaceBox = (index) => {
        const spaceBox = document.createElement('div');
        spaceBox.className = 'space-box';
        spaceBox.dataset.uid = uid;
        spaceBox.dataset.spaceIndex = index;
        spaceBox.title = `Space ${index}`;

        // Get state (default to 'no'/clean)
        const spaceState = state.userAnswers[uid]._text_artifacts_[index.toString()] || 'no';

        // NEW LOGIC: 'no' means clean (no artifact), 'yes' means artifact present
        const isClean = spaceState === 'no';

        spaceBox.classList.add(isClean ? 'correct' : 'incorrect');

        if (isDisabled) {
            spaceBox.classList.add('disabled');
            spaceBox.style.cursor = 'not-allowed';
        } else {
            spaceBox.addEventListener('click', () => {
                const isCurrentlyClean = spaceBox.classList.contains('correct');
                const newCleanState = !isCurrentlyClean;

                spaceBox.classList.remove('correct', 'incorrect');
                spaceBox.classList.add(newCleanState ? 'correct' : 'incorrect');

                setSpaceAnswer(uid, index, newCleanState);
            });
        }
        return spaceBox;
    };

    // Create boxes: Space 0, Word 1, Space 1, Word 2, ..., Space N
    words.forEach((word, index) => {
        // 1. Add Space Box (before word)
        perWordContainer.appendChild(createSpaceBox(index));

        // 2. Add Word Box
        const wordBox = document.createElement('div');
        wordBox.className = 'word-box';
        wordBox.textContent = word;
        wordBox.dataset.uid = uid;
        wordBox.dataset.word = word;

        // UNIQUE KEY STRICT: Use index + word
        const uniqueKey = `${index}_${word}`;
        wordBox.dataset.key = uniqueKey;

        // Set initial state - default to "yes" if not set
        // STRICT MODE: Check unique key only (no backward compatibility)
        let answer = perWordAnswer[uniqueKey];
        if (!answer) answer = 'yes'; // Default

        const isCorrect = answer === 'yes';
        wordBox.classList.add(isCorrect ? 'correct' : 'incorrect');

        if (isDisabled) {
            wordBox.classList.add('disabled');
        } else {
            // Add click handler to toggle state
            wordBox.addEventListener('click', () => {
                const currentState = wordBox.classList.contains('correct');
                const newState = !currentState;

                // Update visual state
                wordBox.classList.remove('correct', 'incorrect');
                wordBox.classList.add(newState ? 'correct' : 'incorrect');

                // Update data using UNIQUE KEY
                setPerWordAnswer(uid, uniqueKey, newState);

                // If marked correct, deselect "No Text" button
                if (newState) {
                    noTextBtn.classList.remove('selected');
                }
            });
        }

        perWordContainer.appendChild(wordBox);
    });

    // 3. Add Final Space Box (after last word)
    perWordContainer.appendChild(createSpaceBox(words.length));

    return true;
}

// Sort annotations so children are placed directly after their parents
function sortAnnotationsByDependency(annotations) {
    const sorted = [];
    const added = new Set();
    const childrenMap = {};
    const topLevel = [];

    // Map children by their first parent UID that exists in the current list
    annotations.forEach(ann => {
        const dependsOn = ann.depends_on || [];
        const parentUid = dependsOn.find(uid => annotations.some(a => a.uid === uid));

        if (parentUid) {
            if (!childrenMap[parentUid]) {
                childrenMap[parentUid] = [];
            }
            childrenMap[parentUid].push(ann);
        } else {
            topLevel.push(ann);
        }
    });

    // Sort topLevel so questions with anchors come first
    topLevel.sort((a, b) => {
        const aHasAnchor = checkIfHasAnchor(a) ? 1 : 0;
        const bHasAnchor = checkIfHasAnchor(b) ? 1 : 0;
        return bHasAnchor - aHasAnchor;
    });

    // Recursively add parent then its children
    function addWithChildren(ann) {
        if (added.has(ann.uid)) return;
        sorted.push(ann);
        added.add(ann.uid);

        const children = childrenMap[ann.uid] || [];
        children.forEach(child => addWithChildren(child));
    }

    // Add all top level annotations (they will pull in their children)
    topLevel.forEach(ann => addWithChildren(ann));

    // Fallback: add any remaining annotations
    annotations.forEach(ann => {
        if (!added.has(ann.uid)) {
            addWithChildren(ann);
        }
    });

    return sorted;
}

// Display questions
function displayQuestions() {
    const container = document.getElementById('questions-container');
    container.innerHTML = '';

    if (!state.currentPrompt || !state.currentPrompt.annotations) {
        container.innerHTML = '<p class="no-questions">No questions available.</p>';
        return;
    }

    const allAnnotations = state.currentPrompt.annotations;

    // Filter annotations based on active annotation modes
    let annotations = allAnnotations.filter(annotation => shouldShowAnnotation(annotation));

    // Sort annotations by dependency so children come right after parents
    annotations = sortAnnotationsByDependency(annotations);

    if (annotations.length === 0) {
        container.innerHTML = '<p class="no-questions">No questions to display for this annotation mode.</p>';
        return;
    }

    annotations.forEach(annotation => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question-item';
        questionDiv.id = `question-item-${annotation.uid}`;

        // Check if this question should be disabled based on dependencies
        const isDisabled = isQuestionDisabled(annotation, allAnnotations);
        if (isDisabled) {
            questionDiv.classList.add('question-disabled');
        }

        // Question content container
        const questionContent = document.createElement('div');
        questionContent.className = 'question-content';

        // Phrase display (if available)
        if (annotation.phrase) {
            const phraseDiv = document.createElement('div');
            phraseDiv.className = 'question-phrase';
            phraseDiv.textContent = `"${annotation.phrase}"`;
            questionContent.appendChild(phraseDiv);
        }

        // Question text with phrase highlighting
        const questionText = document.createElement('div');
        questionText.className = 'question-text';
        questionText.textContent = annotation.question;

        questionContent.appendChild(questionText);

        // ChatGPT indicator (show if using AI answer AND AI is enabled)
        const uid = annotation.uid;
        const userAnswer = state.userAnswers[uid];
        const bqaAiEnabled = isBqaAiEnabled();
        const chatgptAnswer = bqaAiEnabled ? state.chatgptAnswers[uid] : null;
        const isModified = state.modifiedAnswers.has(uid);

        // Show AI indicator if: AI enabled AND there's a chatgpt answer AND (no user answer OR user answer matches chatgpt) AND not modified
        const showAIIndicator = bqaAiEnabled && chatgptAnswer && !isModified && (!userAnswer || userAnswer === chatgptAnswer);

        if (showAIIndicator) {
            const indicator = document.createElement('span');
            indicator.className = 'chatgpt-indicator';
            indicator.title = 'Answer from ChatGPT';
            indicator.id = `chatgpt-indicator-${uid}`;
            questionText.appendChild(indicator);
        }

        // Toggle switch container (will hold both anchor icon and toggle)
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'toggle-container';

        // Check if this is a text rendering question
        const isTextRenderingQuestion = isPerWordTextRendering(annotation);

        // Determine if we should use per-word mode for text rendering
        // Use likert if: it's a text rendering question AND text_likert mode
        const useLikertMode = isTextRenderingQuestion && shouldShowTextLikert();
        // Use per-word if: it's a text rendering question AND (default mode OR text_per_word mode) AND NOT likert mode
        const usePerWordMode = isTextRenderingQuestion && shouldShowTextPerWord() && !shouldShowTextBqa() && !useLikertMode;

        if (useLikertMode) {
            questionDiv.appendChild(questionContent);
            renderTextLikertUI(annotation, questionDiv, isDisabled);
        } else if (usePerWordMode) {
            // Render per-word UI instead of toggle switch
            questionDiv.appendChild(questionContent);
            renderPerWordUI(annotation, questionDiv, isDisabled);
        } else if (shouldShowAnchorLikert() && shouldShowAnnotation(annotation) && checkIfHasAnchor(annotation)) {
            // Anchor Likert Mode
            // Append question content first
            questionDiv.appendChild(questionContent);

            renderAnchorLikertUI(annotation, questionDiv, isDisabled);

        } else {
            // Regular toggle switch for other questions
            // Anchor image indicator (show for specific skill types AND if anchors should be visible)
            const canShowAnchor = shouldShowAnchorImage(annotation) && shouldShowAnchors();
            if (canShowAnchor) {
                const anchorIndicator = document.createElement('span');
                anchorIndicator.className = 'anchor-indicator';
                anchorIndicator.title = 'Hover to see reference image';
                anchorIndicator.dataset.uid = uid;
                anchorIndicator.addEventListener('mouseenter', (e) => showAnchorImage(annotation, e.target));
                anchorIndicator.addEventListener('mouseleave', hideAnchorImage);
                toggleContainer.appendChild(anchorIndicator);
            }

            // Toggle switch
            const toggleDiv = document.createElement('label');
            toggleDiv.className = 'toggle-switch';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `toggle-${annotation.uid}`;
            checkbox.dataset.uid = annotation.uid;
            checkbox.disabled = isDisabled;


            // Set initial value (prefer user answer, then chatgpt if AI enabled, else default to false/no)
            const currentAnswer = userAnswer || chatgptAnswer; // chatgptAnswer is null when AI disabled
            if (currentAnswer === 'yes') {
                checkbox.checked = true;
            }
            // If AI is disabled and no user answer, checkbox stays unchecked (default false)

            // Create "Unsure" button
            const unsureBtn = document.createElement('button');
            unsureBtn.className = 'no-text-btn'; // Reuse the same styling as "No Text" button
            unsureBtn.textContent = 'Unsure';
            unsureBtn.title = 'Mark as unsure';
            unsureBtn.id = `unsure-btn-${uid}`;

            // Check if "unsure" is selected
            if (currentAnswer === 'unsure') {
                unsureBtn.classList.add('selected');
                checkbox.checked = false; // Ensure toggle is off when unsure is selected
                checkbox.disabled = true; // Disable the toggle (gray out)
            }

            if (isDisabled) {
                unsureBtn.disabled = true;
                unsureBtn.style.cursor = 'default';
            } else {
                unsureBtn.addEventListener('click', () => {
                    // Toggle unsure state
                    const isCurrentlyUnsure = unsureBtn.classList.contains('selected');

                    if (isCurrentlyUnsure) {
                        // Deselect unsure - reset to no answer
                        unsureBtn.classList.remove('selected');
                        state.userAnswers[uid] = 'no';
                        checkbox.checked = false;
                        checkbox.disabled = false; // Re-enable toggle
                    } else {
                        // Select unsure
                        state.userAnswers[uid] = 'unsure';
                        state.modifiedAnswers.add(uid);
                        unsureBtn.classList.add('selected');
                        checkbox.checked = false; // Reset toggle
                        checkbox.disabled = true; // Disable toggle (gray out)

                        // Remove AI indicator when manually setting to unsure
                        const indicator = document.getElementById(`chatgpt-indicator-${uid}`);
                        if (indicator) {
                            indicator.remove();
                        }
                    }
                });
            }

            checkbox.addEventListener('change', (e) => {
                const newAnswer = e.target.checked ? 'yes' : 'no';
                state.userAnswers[uid] = newAnswer;

                // Reset "Unsure" button when yes/no is clicked
                unsureBtn.classList.remove('selected');

                // Update AI indicator based on match with chatgpt
                const indicator = document.getElementById(`chatgpt-indicator-${uid}`);
                if (chatgptAnswer && newAnswer === chatgptAnswer) {
                    // Answer matches ChatGPT - show indicator if not already shown
                    if (!indicator) {
                        const newIndicator = document.createElement('span');
                        newIndicator.className = 'chatgpt-indicator';
                        newIndicator.title = 'Answer from ChatGPT';
                        newIndicator.id = `chatgpt-indicator-${uid}`;
                        questionText.appendChild(newIndicator);
                    }
                    state.modifiedAnswers.delete(uid);
                } else {
                    // Answer differs from ChatGPT - remove indicator
                    if (indicator) {
                        indicator.remove();
                    }
                    state.modifiedAnswers.add(uid);
                }

                // Update dependent questions when answer changes
                updateDependentQuestions(uid, annotations);
            });

            const slider = document.createElement('span');
            slider.className = 'slider';

            toggleDiv.appendChild(checkbox);
            toggleDiv.appendChild(slider);

            toggleContainer.appendChild(toggleDiv);
            toggleContainer.appendChild(unsureBtn); // Add unsure button after toggle

            questionDiv.appendChild(questionContent);
            questionDiv.appendChild(toggleContainer);
        }
        container.appendChild(questionDiv);
    });
}

// Escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Show reference image
function showReferenceImage(phrase) {
    const modal = document.getElementById('reference-modal');
    const img = document.getElementById('reference-image');

    // Try to load reference image
    const imagePath = `${API_BASE}/assets/named_entities/${encodeURIComponent(phrase)}.jpg`;
    img.src = imagePath;
    modal.style.display = 'block';

    // Hide if image fails to load
    img.onerror = () => {
        modal.style.display = 'none';
    };
}

// Hide reference image
function hideReferenceImage() {
    const modal = document.getElementById('reference-modal');
    modal.style.display = 'none';
}

// Check if annotation should show anchor image
function shouldShowAnchorImage(annotation) {
    // If no anchor config loaded, don't show anchors
    if (!state.anchorConfig || !state.anchorConfig.annotations) {
        return false;
    }

    // If no current prompt, can't determine prompt_id
    if (!state.currentPrompt) {
        return false;
    }

    const promptId = state.currentPrompt.prompt_id;
    const uid = annotation.uid;

    // Look up this specific annotation in the config
    const configEntry = state.anchorConfig.annotations.find(
        entry => entry.prompt_id === promptId && entry.uid === uid
    );

    // Return true only if config says this annotation needs an anchor
    return configEntry && configEntry.needs_anchor === true;
}

// Global timeout for anchor image hiding
let anchorHideTimeout = null;

// Show anchor image
function showAnchorImage(annotation, iconElement) {
    const modal = document.getElementById('anchor-modal');

    // Clear any pending hide timeout
    if (anchorHideTimeout) {
        clearTimeout(anchorHideTimeout);
        anchorHideTimeout = null;
    }

    if (!state.currentTask || !state.currentPrompt) return;

    // Build path: DATASET_ID / <prompt_id> / <uid>_<index>.png
    const datasetId = state.currentTask.dataset_version;
    const promptId = state.currentPrompt.prompt_id;
    const uid = annotation.uid;

    // Load all 3 anchor images
    for (let i = 1; i <= 3; i++) {
        const img = document.getElementById(`anchor-image-${i}`);
        const imagePath = `${API_BASE}/anchors/${datasetId}/${promptId}/${uid}_${i}.png`;
        img.src = imagePath;

        // Hide image if it fails to load
        img.onerror = () => {
            img.style.display = 'none';
        };

        // Show image on successful load
        img.onload = () => {
            img.style.display = 'block';
        };
    }

    // Position modal next to the icon (to the left)
    if (iconElement) {
        const rect = iconElement.getBoundingClientRect();
        const modalContent = modal.querySelector('.modal-content');
        modal.style.position = 'fixed';
        modal.style.right = `${window.innerWidth - rect.left + 10}px`;
        modal.style.left = 'auto';
        modal.style.top = `${rect.top}px`;
        modal.style.transform = 'none';
    }

    modal.style.display = 'block';

    // Setup modal hover events to keep it visible
    modal.onmouseenter = () => {
        if (anchorHideTimeout) {
            clearTimeout(anchorHideTimeout);
            anchorHideTimeout = null;
        }
    };

    modal.onmouseleave = () => {
        hideAnchorImage();
    };
}

// Hide anchor image with delay
function hideAnchorImage() {
    // Clear any existing timeout
    if (anchorHideTimeout) {
        clearTimeout(anchorHideTimeout);
    }

    // Delay hiding to allow moving mouse to modal
    anchorHideTimeout = setTimeout(() => {
        const modal = document.getElementById('anchor-modal');
        modal.style.display = 'none';
        anchorHideTimeout = null;
    }, 100);
}

// Setup canvas for artifact annotation
function setupCanvas() {
    const img = document.getElementById('main-image');
    const canvas = document.getElementById('artifact-canvas');
    const container = document.querySelector('.image-container');

    // Set canvas size to match image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Position canvas over image
    const imgRect = img.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    canvas.style.width = `${img.offsetWidth}px`;
    canvas.style.height = `${img.offsetHeight}px`;
    canvas.style.left = `${img.offsetLeft}px`;
    canvas.style.top = `${img.offsetTop}px`;

    state.canvas = canvas;
    state.ctx = canvas.getContext('2d');

    // Setup drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
}

// Drawing functions
function startDrawing(e) {
    if (!state.isAnnotatingArtifacts) return;

    state.isDrawing = true;
    const rect = state.canvas.getBoundingClientRect();
    const scaleX = state.canvas.width / rect.width;
    const scaleY = state.canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const brushSize = document.getElementById('brush-size').value;

    // Set drawing properties
    state.ctx.lineWidth = brushSize;
    state.ctx.lineCap = 'round';
    state.ctx.lineJoin = 'round';

    if (state.isEraseMode) {
        state.ctx.globalCompositeOperation = 'destination-out';
        state.ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
    } else {
        state.ctx.globalCompositeOperation = 'source-over';
        state.ctx.strokeStyle = 'rgba(255, 0, 0, 1)';
    }

    state.ctx.beginPath();
    state.ctx.moveTo(x, y);
    state.ctx.lineTo(x, y);
    state.ctx.stroke();
}

function draw(e) {
    if (!state.isDrawing || !state.isAnnotatingArtifacts) return;

    const rect = state.canvas.getBoundingClientRect();
    const scaleX = state.canvas.width / rect.width;
    const scaleY = state.canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    state.ctx.lineTo(x, y);
    state.ctx.stroke();
    state.ctx.beginPath();
    state.ctx.moveTo(x, y);
}

function stopDrawing() {
    if (state.isDrawing) {
        state.ctx.closePath();
        state.isDrawing = false;
    }
}

// Clear canvas
function clearCanvas() {
    if (state.ctx) {
        state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    }
}

// Start artifact annotation - opens modal popup
async function startArtifactAnnotation() {
    // Don't start if artifacts are not allowed in current mode
    if (!shouldShowArtifacts()) return;

    if (!state.currentImage) return;

    state.isAnnotatingArtifacts = true;
    state.isEraseMode = false;

    const modal = document.getElementById('artifact-modal');
    const modalImage = document.getElementById('artifact-modal-image');
    const modalCanvas = document.getElementById('artifact-modal-canvas');

    // Show modal first so we can calculate positions
    modal.style.display = 'flex';

    // Set image source
    modalImage.src = `${API_BASE}/generated_images/${state.currentImage.image_path}`;

    // Wait for image to load before setting up canvas
    modalImage.onload = async () => {
        // Set canvas size to match displayed image size
        const displayWidth = modalImage.clientWidth;
        const displayHeight = modalImage.clientHeight;

        modalCanvas.width = displayWidth;
        modalCanvas.height = displayHeight;
        modalCanvas.style.width = displayWidth + 'px';
        modalCanvas.style.height = displayHeight + 'px';

        // Store canvas reference and scale factors for mask saving
        state.canvas = modalCanvas;
        state.ctx = modalCanvas.getContext('2d');
        state.imageNaturalWidth = modalImage.naturalWidth;
        state.imageNaturalHeight = modalImage.naturalHeight;

        // Setup canvas drawing
        setupModalCanvasDrawing();

        // Load existing mask
        await loadExistingMask();

        // Reset mode toggle
        document.getElementById('artifact-mode-toggle').checked = false;

        // Update brush cursor size
        updateArtifactBrushCursorSize();
    };
}

// Setup canvas drawing for modal
let modalCanvasListenersAttached = false;

function setupModalCanvasDrawing() {
    const canvas = state.canvas;

    // Prevent adding duplicate event listeners
    if (modalCanvasListenersAttached) return;
    modalCanvasListenersAttached = true;

    canvas.addEventListener('mousedown', (e) => {
        if (!state.isAnnotatingArtifacts) return;
        state.isDrawing = true;
        drawOnModal(e);
    });

    canvas.addEventListener('mousemove', (e) => {
        updateArtifactBrushCursorPosition(e);
        if (!state.isDrawing || !state.isAnnotatingArtifacts) return;
        drawOnModal(e);
    });

    canvas.addEventListener('mouseup', () => {
        state.isDrawing = false;
    });

    canvas.addEventListener('mouseleave', () => {
        state.isDrawing = false;
        document.getElementById('artifact-modal-brush-cursor').style.display = 'none';
    });

    canvas.addEventListener('mouseenter', () => {
        if (state.isAnnotatingArtifacts) {
            document.getElementById('artifact-modal-brush-cursor').style.display = 'block';
        }
    });
}

// Draw on modal canvas
function drawOnModal(e) {
    const canvas = state.canvas;
    const ctx = state.ctx;
    const rect = canvas.getBoundingClientRect();
    // Canvas now matches display size, so no scaling needed for drawing
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const brushSize = parseInt(document.getElementById('artifact-brush-size').value);

    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);

    if (state.isEraseMode) {
        ctx.globalCompositeOperation = 'destination-out';
    } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    }

    ctx.fill();
}

// Load existing artifact mask
async function loadExistingMask() {
    if (!state.currentImage || !state.ctx || !state.currentTask) return;

    const image = state.currentImage;
    // Add cache-busting timestamp to prevent loading cached old mask
    const maskPath = `${API_BASE}/annotations/${state.currentTask.id}/artifact_mask/${image.model}/${image.prompt_id}_${state.currentAnnotator}.png?t=${Date.now()}`;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        // Scale mask to fit current canvas size
        state.ctx.drawImage(img, 0, 0, state.canvas.width, state.canvas.height);
    };
    img.onerror = () => {
        // Mask doesn't exist, that's fine
    };
    img.src = maskPath;
}

// Handle mode toggle for artifact modal
function handleArtifactModeToggle() {
    const toggle = document.getElementById('artifact-mode-toggle');
    state.isEraseMode = toggle.checked;
}

// Update brush cursor size for artifact modal
function updateArtifactBrushCursorSize() {
    const brushSize = document.getElementById('artifact-brush-size').value;
    const cursor = document.getElementById('artifact-modal-brush-cursor');
    cursor.style.width = `${brushSize}px`;
    cursor.style.height = `${brushSize}px`;
}

// Update brush cursor position for artifact modal
function updateArtifactBrushCursorPosition(e) {
    const cursor = document.getElementById('artifact-modal-brush-cursor');
    const canvas = state.canvas;
    const container = canvas.parentElement;

    const canvasRect = canvas.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    const x = e.clientX - canvasRect.left;
    const y = e.clientY - canvasRect.top;

    const offsetX = canvasRect.left - containerRect.left;
    const offsetY = canvasRect.top - containerRect.top;

    cursor.style.left = `${offsetX + x}px`;
    cursor.style.top = `${offsetY + y}px`;
}

// Close artifact modal and save
async function closeArtifactModal() {
    state.isAnnotatingArtifacts = false;

    // Hide modal
    document.getElementById('artifact-modal').style.display = 'none';
    document.getElementById('artifact-modal-brush-cursor').style.display = 'none';

    if (!state.currentImage || !state.canvas) return;

    // Check if there's content before saving
    const imageData = state.ctx.getImageData(0, 0, state.canvas.width, state.canvas.height);
    const hasContent = imageData.data.some(channel => channel !== 0);

    // Scale mask to original image dimensions for saving
    const saveCanvas = document.createElement('canvas');
    saveCanvas.width = state.imageNaturalWidth;
    saveCanvas.height = state.imageNaturalHeight;
    const saveCtx = saveCanvas.getContext('2d');
    saveCtx.drawImage(state.canvas, 0, 0, state.imageNaturalWidth, state.imageNaturalHeight);

    const maskData = saveCanvas.toDataURL('image/png');
    const image = state.currentImage;

    await fetch(`${API_BASE}/api/save_artifact_mask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            task_id: state.currentTask.id,
            model: image.model,
            prompt_id: image.prompt_id,
            annotator: state.currentAnnotator,
            mask_data: maskData
        })
    });

    // Update checkbox - user has viewed/checked for artifacts (even if none found)
    document.getElementById('status-artifacts').checked = true;

    // Update mask overlay to show the newly saved mask
    const maskOverlay = document.getElementById('artifact-mask-overlay');
    if (hasContent && image) {
        // Build mask path with cache-busting query string to force reload
        const maskPath = `${API_BASE}/annotations/${state.currentTask.id}/artifact_mask/${image.model}/${image.prompt_id}_${state.currentAnnotator}.png?t=${Date.now()}`;
        maskOverlay.src = maskPath;
    } else {
        maskOverlay.removeAttribute('src');
        maskOverlay.style.display = 'none';
    }
    // Update visibility based on toggle state
    updateMaskVisibility();
}

// Legacy function - kept for compatibility
async function doneAnnotatingArtifacts() {
    await closeArtifactModal();
}

// ============ ARTIFACT POINT ANNOTATION ============

// Render point dots on the main image overlay canvas
function renderPointOverlay() {
    const canvas = document.getElementById('artifact-point-overlay');
    const img = document.getElementById('main-image');
    if (!canvas || !img) return;

    // Size the overlay canvas to match the displayed image
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    canvas.style.width = img.clientWidth + 'px';
    canvas.style.height = img.clientHeight + 'px';
    canvas.style.left = img.offsetLeft + 'px';
    canvas.style.top = img.offsetTop + 'px';

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.artifactPoints || state.artifactPoints.length === 0) return;

    // Scale from image-space to display-space
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    const radius = 6;

    for (const pt of state.artifactPoints) {
        const dx = pt.x * scaleX;
        const dy = pt.y * scaleY;

        ctx.beginPath();
        ctx.arc(dx, dy, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Open artifact point modal
async function startArtifactPointAnnotation() {
    if (!shouldShowArtifactsPoint()) return;
    if (!state.currentImage) return;

    state.isPointRemoveMode = false;

    const modal = document.getElementById('artifact-point-modal');
    const modalImage = document.getElementById('artifact-point-modal-image');
    const modalCanvas = document.getElementById('artifact-point-modal-canvas');

    modal.style.display = 'flex';
    modalImage.src = `${API_BASE}/generated_images/${state.currentImage.image_path}`;

    modalImage.onload = () => {
        const displayWidth = modalImage.clientWidth;
        const displayHeight = modalImage.clientHeight;

        modalCanvas.width = displayWidth;
        modalCanvas.height = displayHeight;
        modalCanvas.style.width = displayWidth + 'px';
        modalCanvas.style.height = displayHeight + 'px';

        // Reset mode toggle
        document.getElementById('artifact-point-mode-toggle').checked = false;

        // Render existing points
        renderModalPoints(modalCanvas, modalImage);
    };
}

// Render points on the modal canvas
function renderModalPoints(canvas, img) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.artifactPoints || state.artifactPoints.length === 0) return;

    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;
    const radius = 8;

    for (const pt of state.artifactPoints) {
        const dx = pt.x * scaleX;
        const dy = pt.y * scaleY;

        ctx.beginPath();
        ctx.arc(dx, dy, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Handle click on the modal canvas (add or remove point)
let pointModalListenersAttached = false;

function setupPointModalListeners() {
    const canvas = document.getElementById('artifact-point-modal-canvas');
    if (pointModalListenersAttached) return;
    pointModalListenersAttached = true;

    canvas.addEventListener('click', (e) => {
        const img = document.getElementById('artifact-point-modal-image');
        const rect = canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Convert to image-space coordinates
        const imgX = (clickX / canvas.width) * img.naturalWidth;
        const imgY = (clickY / canvas.height) * img.naturalHeight;

        if (state.isPointRemoveMode) {
            // Remove nearest point within threshold
            const threshold = 20; // pixels in display space
            let minDist = Infinity;
            let minIdx = -1;

            for (let i = 0; i < state.artifactPoints.length; i++) {
                const pt = state.artifactPoints[i];
                const scaleX = canvas.width / img.naturalWidth;
                const scaleY = canvas.height / img.naturalHeight;
                const dx = pt.x * scaleX - clickX;
                const dy = pt.y * scaleY - clickY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minDist) {
                    minDist = dist;
                    minIdx = i;
                }
            }

            if (minIdx >= 0 && minDist < threshold) {
                state.artifactPoints.splice(minIdx, 1);
            }
        } else {
            // Add point
            state.artifactPoints.push({ x: Math.round(imgX), y: Math.round(imgY) });
        }

        renderModalPoints(canvas, img);
    });

    // Cursor style changes
    canvas.style.cursor = 'crosshair';
}

// Close artifact point modal and save
async function closeArtifactPointModal() {
    const modal = document.getElementById('artifact-point-modal');
    modal.style.display = 'none';

    if (!state.currentImage || !state.currentTask) return;

    const image = state.currentImage;

    // Save points to server (convert {x,y} objects to [x,y] tuples)
    const tuples = state.artifactPoints.map(p => [p.x, p.y]);
    await fetch(`${API_BASE}/api/save_artifact_points`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            task_id: state.currentTask.id,
            model: image.model,
            prompt_id: image.prompt_id,
            annotator: state.currentAnnotator,
            points: tuples
        })
    });

    // Update artifact status checkbox
    document.getElementById('status-artifacts').checked = true;

    // Update point overlay on main image
    renderPointOverlay();
}

// Clear all artifact points
function clearAllArtifactPoints() {
    state.artifactPoints = [];
    const canvas = document.getElementById('artifact-point-modal-canvas');
    const img = document.getElementById('artifact-point-modal-image');
    if (canvas && img) {
        renderModalPoints(canvas, img);
    }
}

// Save and go to next image/prompt
async function saveAndNext() {
    if (!state.currentImage || !state.currentPrompt) {
        // No image, just move to next prompt
        goToNextPrompt();
        return;
    }

    // Validation: Check if aesthetics rating is set
    // DISABLED FOR DEVELOPMENT
    // if (state.aestheticsRating === 0) {
    //     alert('Please rate the aesthetics of the image before proceeding.');
    //     return;
    // }

    // Validation: Check if artifacts have been checked (user opened artifact mode)
    // DISABLED FOR DEVELOPMENT
    // const hasCheckedArtifacts = document.getElementById('status-artifacts').checked;
    // if (!hasCheckedArtifacts) {
    //     alert('Please click "Artifacts Annotation Mode" to check the image for artifacts before proceeding (even if there are none).');
    //     return;
    // }

    try {
        // Collect answers only for annotations that are shown based on active annotation modes
        const allAnswers = {};
        const annotations = state.currentPrompt.annotations || [];

        for (const annotation of annotations) {
            // Only save annotations that are actually shown based on the current task's annotation modes
            if (!shouldShowAnnotation(annotation)) {
                continue;
            }
            const uid = annotation.uid;

            // Check if this is a text likert question
            const isTextRendering = annotation.skill === 'text_rendering' &&
                (annotation.subskill === 'rendering_accuracy' || annotation.subskill === 'numerical');
            const isTextLikertMode = state.activeAnnotationModes && state.activeAnnotationModes.includes('text_likert');

            // Use userAnswer if set (even if 0), otherwise use chatgptAnswer
            let val = state.userAnswers[uid];
            if (val === undefined || val === null) {
                val = state.chatgptAnswers[uid];
            }

            // Default to 'no' ONLY if not in text likert mode (where explicit rating is required)
            // And val is not set
            if ((val === undefined || val === null || val === '') && !(isTextRendering && isTextLikertMode)) {
                val = 'no';
            }

            if (val !== undefined && val !== null && val !== '') {
                if (isTextRendering && isTextLikertMode) {
                    allAnswers['_text_likert'] = val;
                } else {
                    allAnswers[uid] = val;
                }
            }
        }

        // Add aesthetics rating to answers (only if shown and > 0)
        if (shouldShowAesthetics() && state.aestheticsRating > 0) {
            allAnswers._aesthetics_rating = state.aestheticsRating;
        }

        // Save annotations (only if we have answers to save)
        if (Object.keys(allAnswers).length > 0) {
            await fetch(`${API_BASE}/api/save_annotations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    task_id: state.currentTask.id,
                    model: state.currentImage.model,
                    prompt_id: state.currentImage.prompt_id,
                    annotator: state.currentAnnotator,
                    answers: allAnswers
                })
            });

            // Mark current image as annotated and update completion cache
            state.currentImage.annotated = true;
            state.currentImage._completionCached = true;

            // Update VQA checkbox
            document.getElementById('status-vqa').checked = true;
        }

        // Clear canvas and exit artifact mode if active
        if (state.isAnnotatingArtifacts) {
            clearCanvas();
            await doneAnnotatingArtifacts();
        }

        // Move to next image in the list
        state.allImagesIndex++;
        if (state.allImagesIndex >= state.allImages.length) {
            // Completed all images, loop back
            state.allImagesIndex = 0;
            console.log('Completed all images, looping back to start');
        }
        displayCurrentListImage();
    } catch (error) {
        console.error('Error saving annotations:', error);
        alert('Error saving annotations. Please try again.');
    }
}

// Go to next prompt
async function goToNextPrompt() {
    const startIndex = state.currentPromptIndex;
    let nextIndex = state.currentPromptIndex;

    // Find next prompt with relevant annotations
    do {
        nextIndex++;
        if (nextIndex >= state.prompts.length) {
            nextIndex = 0; // Loop back to start
        }

        // If we've checked all prompts and none have relevant annotations, stay on current
        if (nextIndex === startIndex) {
            console.log('No more prompts with relevant annotations');
            return;
        }
    } while (!promptHasRelevantAnnotations(state.prompts[nextIndex]));

    state.currentPromptIndex = nextIndex;
    state.currentPrompt = state.prompts[state.currentPromptIndex];
    await loadCurrentPromptImages();
    displayCurrentImage();
}

// Go to previous image
async function goBack() {
    // Exit artifact mode if active
    if (state.isAnnotatingArtifacts) {
        clearCanvas();
        await doneAnnotatingArtifacts();
    }

    // Go to previous in images list
    if (state.allImagesIndex > 0) {
        state.allImagesIndex--;
        displayCurrentListImage();
    }
}

// Update progress
async function updateProgress() {
    // Filter prompts to only those with relevant annotations for current mode
    const relevantPrompts = state.prompts.filter(p => promptHasRelevantAnnotations(p));
    const totalPrompts = relevantPrompts.length;

    // Find current position within relevant prompts
    const currentRelevantIndex = relevantPrompts.findIndex(p => p.prompt_id === state.currentPrompt?.prompt_id);

    // Update back button state - disabled only at very beginning of relevant prompts
    const backBtn = document.getElementById('btn-back');
    backBtn.disabled = (currentRelevantIndex <= 0 && state.currentImageIndex === 0);

    // Count annotated prompts by checking actual annotation status (only relevant ones)
    let annotatedPrompts = 0;

    for (let i = 0; i < relevantPrompts.length; i++) {
        const prompt = relevantPrompts[i];
        const isComplete = await checkPromptCompletion(prompt.prompt_id);
        if (isComplete) {
            annotatedPrompts++;
        }
    }

    // Calculate percentage based on annotated prompts
    const percentage = totalPrompts > 0 ? (annotatedPrompts / totalPrompts) * 100 : 0;

    // Update header progress bar - show position in relevant prompts
    document.getElementById('header-progress-fill').style.width = `${percentage}%`;
    document.getElementById('header-progress-text').textContent = `${annotatedPrompts} / ${totalPrompts}`;
}

// Check if a prompt is fully annotated based on active annotation modes
async function checkPromptCompletion(promptId) {
    if (!state.currentTask) return false;

    // Determine which checks are required based on active annotation modes
    const modes = state.activeAnnotationModes;
    const checkAll = !modes || modes.length === 0;

    // VQA (BQA or text) is required if any of these modes are active
    const requireVqa = checkAll || shouldShowBqaQuestions() || shouldShowTextRendering();

    // Artifacts are only required if artifacts_brush mode is active
    const requireArtifacts = checkAll || shouldShowArtifacts();

    // Artifact points are required if artifacts_point mode is active
    const requireArtifactPoints = shouldShowArtifactsPoint();

    // Aesthetics are only required if artifacts_likert mode is active
    const requireAesthetics = checkAll || shouldShowAesthetics();

    try {
        // Get images for this prompt
        const response = await fetch(
            `${API_BASE}/api/task/${state.currentTask.id}/images_for_prompt/${promptId}?annotator=${state.currentAnnotator}&t=${Date.now()}`
        );
        const data = await response.json();
        const images = data.images || [];

        if (images.length === 0) return false;

        // Check if all images have all required annotations
        for (const image of images) {
            // Check VQA annotations (only if required)
            if (requireVqa && !image.annotated) {
                return false;
            }

            // Only fetch detailed annotation data if we need artifacts or aesthetics or points
            if (requireArtifacts || requireAesthetics || requireArtifactPoints) {
                const annotResponse = await fetch(
                    `${API_BASE}/api/image/${state.currentTask.id}/${image.model}/${image.prompt_id}?annotator=${state.currentAnnotator}&t=${Date.now()}`
                );
                const annotData = await annotResponse.json();

                // Check artifacts brush (only if required)
                if (requireArtifacts) {
                    const hasArtifacts = annotData.has_artifact_mask;
                    if (!hasArtifacts) {
                        return false;
                    }
                }

                // Check artifact points (only if required)
                if (requireArtifactPoints) {
                    if (!annotData.has_artifact_points) {
                        return false;
                    }
                }

                // Check aesthetics (only if required)
                if (requireAesthetics) {
                    const hasAesthetics = annotData.user_annotations && annotData.user_annotations._aesthetics_rating > 0;
                    if (!hasAesthetics) {
                        return false;
                    }
                }
            }
        }

        return true;
    } catch (error) {
        console.error(`Error checking prompt ${promptId}:`, error);
        return false;
    }
}

// Check if a single image is fully annotated based on active annotation modes
async function checkSingleImageCompletion(image) {
    if (!state.currentTask) return false;

    // Determine which checks are required based on active annotation modes
    const modes = state.activeAnnotationModes;
    const checkAll = !modes || modes.length === 0;

    // VQA (BQA or text) is required if any of these modes are active
    const requireVqa = checkAll || shouldShowBqaQuestions() || shouldShowTextRendering();

    // Artifacts are only required if artifacts_brush mode is active
    const requireArtifacts = checkAll || shouldShowArtifacts();

    // Artifact points are required if artifacts_point mode is active
    const requireArtifactPoints = shouldShowArtifactsPoint();

    // Aesthetics are only required if artifacts_likert mode is active
    const requireAesthetics = checkAll || shouldShowAesthetics();

    // Check VQA (only if required)
    if (requireVqa && !image.annotated) {
        return false;
    }

    // Only fetch detailed annotation data if we need artifacts or aesthetics or points
    if (requireArtifacts || requireAesthetics || requireArtifactPoints) {
        try {
            const annotResponse = await fetch(
                `${API_BASE}/api/image/${state.currentTask.id}/${image.model}/${image.prompt_id}?annotator=${state.currentAnnotator}&t=${Date.now()}`
            );
            const annotData = await annotResponse.json();

            // Check artifacts brush (only if required)
            if (requireArtifacts) {
                const hasArtifacts = annotData.has_artifact_mask;
                if (!hasArtifacts) {
                    return false;
                }
            }

            // Check artifact points (only if required)
            if (requireArtifactPoints) {
                if (!annotData.has_artifact_points) {
                    return false;
                }
            }

            // Check aesthetics (only if required)
            if (requireAesthetics) {
                const hasAesthetics = annotData.user_annotations && annotData.user_annotations._aesthetics_rating > 0;
                if (!hasAesthetics) {
                    return false;
                }
            }
        } catch (error) {
            console.error(`Error checking image completion:`, error);
            return false;
        }
    }

    return true;
}

// Toggle progress dropdown
function toggleProgressDropdown() {
    const dropdown = document.getElementById('progress-dropdown');
    const isVisible = dropdown.style.display !== 'none';

    if (isVisible) {
        dropdown.style.display = 'none';
    } else {
        updateProgressDropdown();
        dropdown.style.display = 'block';
    }
}

// Store dropdown data for filtering
let dropdownData = [];

// Create anonymous model name mapping
function getAnonymousModelName(model) {
    if (!state.modelNameMap) {
        state.modelNameMap = {};
        const models = state.currentTask?.models || [];
        models.forEach((m, i) => {
            state.modelNameMap[m] = `Model ${i + 1}`;
        });
    }
    return state.modelNameMap[model] || model;
}

// Update progress dropdown with image list
async function updateProgressDropdown() {
    const dropdownList = document.getElementById('progress-dropdown-list');
    dropdownList.innerHTML = '<div style="padding: 20px; text-align: center; color: #6b7280;">Loading...</div>';

    // Clear search input
    document.getElementById('progress-search').value = '';

    dropdownData = [];

    // Always show individual images from allImages list
    for (let i = 0; i < state.allImages.length; i++) {
        const item = state.allImages[i];
        const isComplete = await checkSingleImageCompletion(item.image);
        dropdownData.push({
            index: i,
            promptId: item.prompt.prompt_id,
            isComplete: isComplete
        });
    }

    // Render the dropdown items
    renderDropdownItems(dropdownData);
}

// Render dropdown items (used for initial load and filtering)
function renderDropdownItems(items) {
    const dropdownList = document.getElementById('progress-dropdown-list');
    dropdownList.innerHTML = '';

    if (items.length === 0) {
        dropdownList.innerHTML = '<div style="padding: 20px; text-align: center; color: #6b7280;">No matching items</div>';
        return;
    }

    for (const item of items) {
        const i = item.index;
        const isActive = (i === state.allImagesIndex);
        const isCompleted = item.isComplete;

        const itemDiv = document.createElement('div');
        itemDiv.className = `progress-dropdown-item ${isActive ? 'active' : ''}`;

        // Display text: just prompt ID (no model indicator to avoid bias)
        const displayText = `Prompt ${item.promptId}`;

        itemDiv.innerHTML = `
            <span class="progress-status-icon ${isCompleted ? 'completed' : 'incomplete'}">
                ${isCompleted ? '✓' : '○'}
            </span>
            <span class="progress-item-index">${i + 1}.</span>
            <span class="progress-item-text">${displayText}</span>
        `;

        itemDiv.addEventListener('click', async () => {
            // Navigate to this image in the list
            state.allImagesIndex = i;
            await displayCurrentListImage();
            document.getElementById('progress-dropdown').style.display = 'none';
        });

        dropdownList.appendChild(itemDiv);
    }
}

// Filter dropdown items by prompt ID
function filterDropdownItems(searchTerm) {
    const filtered = dropdownData.filter(item =>
        String(item.promptId).toLowerCase().includes(searchTerm.toLowerCase())
    );
    renderDropdownItems(filtered);
}

// Close dropdown when clicking outside
function handleClickOutside(event) {
    const progressBar = document.getElementById('header-progress');
    const dropdown = document.getElementById('progress-dropdown');

    if (!progressBar.contains(event.target) && dropdown.style.display !== 'none') {
        dropdown.style.display = 'none';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Start annotation button in modal
    document.getElementById('start-annotation-btn').addEventListener('click', startAnnotation);

    // Task selection change - show task details
    document.getElementById('task-select').addEventListener('change', (e) => {
        const task = state.tasks.find(t => t.id === e.target.value);
        if (task) showTaskDetails(task);
    });

    // Progress dropdown search
    document.getElementById('progress-search').addEventListener('input', (e) => {
        filterDropdownItems(e.target.value);
    });

    // Navigation buttons
    document.getElementById('btn-back').addEventListener('click', goBack);
    document.getElementById('btn-next').addEventListener('click', saveAndNext);

    // Artifact annotation - click on brush button or image
    document.getElementById('artifact-brush-button').addEventListener('click', startArtifactAnnotation);

    // Mask toggle button
    document.getElementById('mask-toggle-button').addEventListener('click', toggleMaskVisibility);
    document.getElementById('main-image').addEventListener('click', () => {
        if (shouldShowArtifactsPoint()) {
            startArtifactPointAnnotation();
        } else {
            startArtifactAnnotation();
        }
    });

    // Artifact modal controls
    document.getElementById('artifact-done-btn').addEventListener('click', closeArtifactModal);
    document.getElementById('artifact-clear-canvas').addEventListener('click', clearCanvas);
    document.getElementById('artifact-mode-toggle').addEventListener('change', handleArtifactModeToggle);
    document.getElementById('artifact-brush-size').addEventListener('input', updateArtifactBrushCursorSize);

    // Artifact point button
    document.getElementById('artifact-point-button').addEventListener('click', startArtifactPointAnnotation);

    // Artifact point modal controls
    document.getElementById('artifact-point-done-btn').addEventListener('click', closeArtifactPointModal);
    document.getElementById('artifact-point-clear-btn').addEventListener('click', clearAllArtifactPoints);
    document.getElementById('artifact-point-mode-toggle').addEventListener('change', () => {
        state.isPointRemoveMode = document.getElementById('artifact-point-mode-toggle').checked;
    });
    setupPointModalListeners();

    // Click outside artifact point modal content to close
    document.getElementById('artifact-point-modal').addEventListener('click', (e) => {
        if (e.target.id === 'artifact-point-modal') {
            closeArtifactPointModal();
        }
    });

    // Click outside artifact modal content to close
    document.getElementById('artifact-modal').addEventListener('click', (e) => {
        if (e.target.id === 'artifact-modal') {
            closeArtifactModal();
        }
    });

    // Close reference modal on click
    document.getElementById('reference-modal').addEventListener('click', hideReferenceImage);

    // Batch VQA button
    document.getElementById('btn-batch-vqa').addEventListener('click', startBatchVQA);
    document.getElementById('batch-vqa-cancel-btn').addEventListener('click', cancelBatchVQA);

    // Star rating
    setupStarRating();

    // Header progress bar click - toggle dropdown
    document.getElementById('header-progress').addEventListener('click', (e) => {
        e.stopPropagation();
        toggleProgressDropdown();
    });

    // Prevent closing when interacting with the dropdown (like the search box)
    document.getElementById('progress-dropdown').addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', handleClickOutside);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        // Ignore if focus is on an input or textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        // Ignore if modifier keys are pressed
        if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) {
            return;
        }

        if (e.key === 'ArrowRight') {
            e.preventDefault(); // Prevent scrolling
            document.getElementById('btn-next').click();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault(); // Prevent scrolling
            document.getElementById('btn-back').click();
        }
    });
}

// Setup star rating interaction
function setupStarRating() {
    const ratingContainer = document.getElementById('aesthetics-rating');
    const stars = ratingContainer.querySelectorAll('.star');

    stars.forEach(star => {
        // Click to select rating
        star.addEventListener('click', () => {
            const rating = parseInt(star.dataset.rating);
            state.aestheticsRating = rating;
            updateStarDisplay(rating);
            // Update aesthetics checkbox
            document.getElementById('status-aesthetics').checked = rating > 0;
        });

        // Hover effect
        star.addEventListener('mouseenter', () => {
            const rating = parseInt(star.dataset.rating);
            highlightStars(rating);
        });
    });

    // Reset hover on mouse leave
    ratingContainer.addEventListener('mouseleave', () => {
        updateStarDisplay(state.aestheticsRating);
    });
}

// Highlight stars up to rating
function highlightStars(rating) {
    const ratingContainer = document.getElementById('aesthetics-rating');
    const stars = ratingContainer.querySelectorAll('.star');
    stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('hover');
        } else {
            star.classList.remove('hover');
        }
    });
}

// Update star display based on selected rating
function updateStarDisplay(rating) {
    const ratingContainer = document.getElementById('aesthetics-rating');
    const stars = ratingContainer.querySelectorAll('.star');
    stars.forEach((star, index) => {
        star.classList.remove('hover');
        if (index < rating) {
            star.classList.add('selected');
        } else {
            star.classList.remove('selected');
        }
    });
}

// ============ BATCH VQA PROCESSING ============

async function startBatchVQA() {
    if (!state.currentTask || state.prompts.length === 0) {
        alert('Please select a task and load prompts first');
        return;
    }

    if (state.batchVqaRunning) {
        alert('Batch processing is already running');
        return;
    }

    // Show modal
    const modal = document.getElementById('batch-vqa-modal');
    modal.style.display = 'flex';

    updateBatchVQAUI('Scanning for images to process...', 0, 0);

    try {
        // Start batch processing - get list of images to process
        const startResponse = await fetch(`${API_BASE}/api/batch_vqa/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: state.currentTask.id,
                prompts: state.prompts
            })
        });

        const startData = await startResponse.json();

        if (startData.error) {
            alert(`Error: ${startData.error}`);
            closeBatchVQAModal();
            return;
        }

        if (startData.total === 0) {
            updateBatchVQAUI('All images already have AI annotations!', 0, 0);
            document.getElementById('batch-vqa-cancel-btn').textContent = 'Close';
            await fetch(`${API_BASE}/api/batch_vqa/reset`, { method: 'POST' });
            return;
        }

        // Build queue of images to process
        state.batchVqaQueue = await buildBatchVQAQueue();
        state.batchVqaRunning = true;
        state.batchVqaCancelled = false;

        // Process images one by one
        await processBatchVQAQueue();

    } catch (error) {
        console.error('Error starting batch VQA:', error);
        alert(`Error: ${error.message}`);
        closeBatchVQAModal();
    }
}

async function buildBatchVQAQueue() {
    const queue = [];
    const task = state.currentTask;
    const models = task.models || null;

    for (const promptData of state.prompts) {
        const promptId = promptData.prompt_id;
        if (!promptId) continue;

        // Get images for this prompt
        try {
            const response = await fetch(
                `${API_BASE}/api/task/${task.id}/images_for_prompt/${promptId}?annotator=${state.currentAnnotator}`
            );
            const data = await response.json();

            for (const img of data.images) {
                // Check if chatgpt annotations exist
                const annotResponse = await fetch(
                    `${API_BASE}/api/image/${task.id}/${img.model}/${img.prompt_id}?annotator=${state.currentAnnotator}`
                );
                const annotData = await annotResponse.json();

                if (!annotData.has_chatgpt) {
                    queue.push({
                        prompt_data: promptData,
                        image: img
                    });
                }
            }
        } catch (error) {
            console.error(`Error checking images for prompt ${promptId}:`, error);
        }
    }

    return queue;
}

async function processBatchVQAQueue() {
    const total = state.batchVqaQueue.length;
    state.batchVqaProcessed = 0;
    state.batchVqaSucceeded = 0;
    state.batchVqaFailed = 0;
    const CONCURRENCY = 10; // Process 10 images in parallel

    // Process in batches of CONCURRENCY
    for (let i = 0; i < state.batchVqaQueue.length; i += CONCURRENCY) {
        if (state.batchVqaCancelled) {
            updateBatchVQAUI('Cancelled', state.batchVqaProcessed, total);
            break;
        }

        const batch = state.batchVqaQueue.slice(i, i + CONCURRENCY);

        // Process batch in parallel, updating progress as each completes
        const promises = batch.map(async (item) => {
            if (state.batchVqaCancelled) return { cancelled: true };

            const imageName = `${item.image.model}/${item.image.prompt_id}`;
            document.getElementById('batch-vqa-current').textContent = `Processing: ${imageName}`;

            try {
                const response = await fetch(`${API_BASE}/api/batch_vqa/process_next`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        task_id: state.currentTask.id,
                        prompt_data: item.prompt_data,
                        image: item.image,
                        dataset_version: state.currentTask.dataset_version,
                        preprocess_with_al: false
                    })
                });
                const data = await response.json();

                // Update progress immediately when each request completes
                state.batchVqaProcessed++;
                if (data.success) {
                    state.batchVqaSucceeded++;
                } else {
                    state.batchVqaFailed++;
                }
                updateBatchVQAUI(`Processing... (${state.batchVqaSucceeded} ✓, ${state.batchVqaFailed} ✗)`, state.batchVqaProcessed, total);

                return data;
            } catch (error) {
                console.error(`Error processing ${item.image.prompt_id}:`, error);
                state.batchVqaProcessed++;
                state.batchVqaFailed++;
                updateBatchVQAUI(`Processing... (${state.batchVqaSucceeded} ✓, ${state.batchVqaFailed} ✗)`, state.batchVqaProcessed, total);
                return { error: error.message };
            }
        });

        const results = await Promise.all(promises);

        // Check for cancellation
        if (results.some(r => r.cancelled)) {
            updateBatchVQAUI('Cancelled', state.batchVqaProcessed, total);
            break;
        }
    }

    // Done
    state.batchVqaRunning = false;
    await fetch(`${API_BASE}/api/batch_vqa/reset`, { method: 'POST' });

    if (!state.batchVqaCancelled) {
        updateBatchVQAUI(`Complete! (${state.batchVqaSucceeded} ✓, ${state.batchVqaFailed} ✗)`, state.batchVqaProcessed, total);
    }

    document.getElementById('batch-vqa-cancel-btn').textContent = 'Close';
    document.getElementById('batch-vqa-current').textContent = 'Done';

    // Refresh current image annotations
    if (state.currentImage) {
        await loadImageAnnotations();
        displayQuestions();
    }
}

function updateBatchVQAUI(statusText, processed, total) {
    document.getElementById('batch-vqa-status-text').textContent = statusText;
    document.getElementById('batch-vqa-count').textContent = `${processed} / ${total}`;

    const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;
    document.getElementById('batch-vqa-percentage').textContent = `${percentage}%`;
    document.getElementById('batch-vqa-progress-fill').style.width = `${percentage}%`;
}

async function cancelBatchVQA() {
    if (state.batchVqaRunning) {
        state.batchVqaCancelled = true;
        await fetch(`${API_BASE}/api/batch_vqa/cancel`, { method: 'POST' });
        document.getElementById('batch-vqa-status-text').textContent = 'Cancelling...';
    } else {
        closeBatchVQAModal();
    }
}

function closeBatchVQAModal() {
    document.getElementById('batch-vqa-modal').style.display = 'none';
    document.getElementById('batch-vqa-cancel-btn').textContent = 'Cancel';
    state.batchVqaRunning = false;
    state.batchVqaCancelled = false;
    state.batchVqaQueue = [];
}

// Toggle mask visibility
function toggleMaskVisibility() {
    state.maskVisible = !state.maskVisible;
    updateMaskVisibility();
}

// Update mask visibility based on state
function updateMaskVisibility() {
    const maskOverlay = document.getElementById('artifact-mask-overlay');
    const pointOverlay = document.getElementById('artifact-point-overlay');
    const toggleButton = document.getElementById('mask-toggle-button');

    if (state.maskVisible) {
        // Show mask if it exists and is loaded
        const src = maskOverlay.getAttribute('src');
        if (src && src.length > 0) {
            maskOverlay.style.display = 'block';
        }
        // Show point overlay
        if (pointOverlay) {
            pointOverlay.style.display = 'block';
        }
        toggleButton.classList.add('active');
        toggleButton.classList.remove('inactive');
    } else {
        // Hide mask and points
        maskOverlay.style.display = 'none';
        if (pointOverlay) {
            pointOverlay.style.display = 'none';
        }
        toggleButton.classList.remove('active');
        toggleButton.classList.add('inactive');
    }
}

// Sync right panel height with left panel
function syncPanelHeights() {
    const leftPanel = document.querySelector('.left-panel');
    const rightPanel = document.querySelector('.right-panel');
    if (leftPanel && rightPanel) {
        rightPanel.style.maxHeight = leftPanel.offsetHeight + 'px';
    }
}

// Setup resize observer for panel height sync
const resizeObserver = new ResizeObserver(() => {
    syncPanelHeights();
});

// Start app
init();

// Observe left panel for size changes
const leftPanel = document.querySelector('.left-panel');
if (leftPanel) {
    resizeObserver.observe(leftPanel);
}
window.addEventListener('resize', syncPanelHeights);
