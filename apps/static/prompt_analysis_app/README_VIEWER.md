# T2I Prompts Skill Analyzer

A modern web viewer for analyzing text-to-image prompts tagged with skills and subskills. Features a futuristic data analytics interface with filtering, detailed analysis, and histogram visualizations.

## Features

- **Dynamic Filtering**: Filter prompts by skills and subskills
- **Detailed Analysis**: View skill breakdown with associated phrases for each prompt
- **Analytics Dashboard**: Visualize skill and subskill occurrence with interactive histograms
- **Modern UI**: Futuristic data analytics design with smooth animations and responsive layout
- **Theme Toggle**: Switch between dark and light themes with persistent preference
- **Real-time Updates**: Dynamic skill taxonomy loaded from `skills.py`

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask backend:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
image-fine-eval/
├── app.py                          # Flask backend API
├── skills.py                       # Skill taxonomy definitions
├── requirements.txt                # Python dependencies
├── static/
│   ├── index.html                 # Main HTML page
│   ├── styles.css                 # Futuristic styling
│   └── app.js                     # Frontend JavaScript
└── assets/
    ├── prompts/
    │   └── prompt_v1.1.txt        # Tagging prompt documentation
    └── v1.1-gpt-5-mini/           # Tagged prompt JSON files
        ├── 0.json
        ├── 1.json
        └── ...
```

## API Endpoints

- `GET /api/skills` - Get skill taxonomy
- `GET /api/prompts` - Get all prompts (supports `?skill=` and `?subskill=` filters)
- `GET /api/prompts/<id>` - Get specific prompt by ID
- `GET /api/statistics` - Get overall statistics
- `GET /api/histogram?type=skills|subskills` - Get histogram data

## Usage

### Filtering Prompts
1. Select a skill from the dropdown to filter prompts
2. If the skill has subskills, select a specific subskill (optional)
3. Click "Clear Filters" to reset

### Viewing Prompt Details
1. Click on any prompt card in the list
2. The detail panel shows:
   - Full prompt text
   - Metadata (ID, type, model)
   - Skill breakdown with associated phrases

### Analytics
1. Toggle between "Skills" and "Subskills" in the histogram section
2. View the top 15 most frequent skills/subskills
3. Hover over bars for exact counts

### Theme Switching
1. Click the theme toggle button (🌙/☀️) in the header
2. Switch between dark and light themes
3. Your preference is automatically saved to browser storage

## Customization

### Updating Skills
Edit `skills.py` to modify the skill taxonomy. The web viewer will automatically load the updated taxonomy on restart.

### Styling
Modify `static/styles.css` to customize the appearance. The design uses CSS variables for easy theme customization:

```css
:root {
    --primary-bg: #0a0e27;
    --accent-blue: #00d4ff;
    --accent-purple: #a855f7;
    /* ... more variables */
}
```

## Data Format

Prompt JSON files should follow this structure:

```json
{
  "prompt_id": 0,
  "prompt": "Your prompt text here",
  "prompt_type": "Gecko(S)",
  "dataset_id": "dataset/name",
  "prompt_version": "v1.1",
  "llm_model": "gpt-5-mini",
  "skills": ["skill1:subskill1", "skill2"],
  "phrases": [["phrase1", "phrase2"], ["phrase3"]]
}
```

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers (responsive design)

## Troubleshooting

**Port already in use**: Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

**CORS errors**: Ensure `flask-cors` is installed and the frontend is accessing the correct API URL.

**No prompts showing**: Verify JSON files exist in `assets/v1.1-gpt-5-mini/` directory.
