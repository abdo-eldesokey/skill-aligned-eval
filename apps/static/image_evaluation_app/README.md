# Image Generation Evaluation Tool

A Flask-based web application for evaluating and annotating generated images with Visual Question Answering (VQA) and artifact detection.

## Features

1. **Multi-Annotator Support**: Multiple annotators can evaluate images independently
2. **VQA Annotation**: Answer yes/no questions about image content based on skill taxonomy
3. **ChatGPT Integration**: Automatically generates initial annotations using GPT-4o-mini
4. **Named Entity References**: Hover over named entities to see reference images
5. **Artifact Annotation**: Draw on images to mark visual artifacts
6. **Progress Tracking**: Track annotation progress across all images
7. **Smart Navigation**: Prioritizes unannotated images

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key:
```bash
# Create .env file
OPENAI_API_KEY=your_api_key_here
```

3. Configure annotators in `assets/annotators.json`

4. Run the application:
```bash
python run_image_evaluation_app.py
```

5. Open browser to `http://localhost:5001`

## Directory Structure

```
assets/
├── annotators.json              # List of annotators
├── generated_images/            # Generated images organized by model
│   └── <model-name>/
│       └── <dataset-id>/
│           ├── imgs/            # Source images
│           ├── vqa/             # VQA annotations per annotator
│           └── artifact_masks/  # Artifact masks per annotator
├── generation_prompts/          # Prompt data with questions
│   └── <dataset-id>/
│       └── <prompt-id>.json
└── named_entities/              # Reference images for named entities
```

## Usage

1. **Select Annotator**: Choose your name from the dropdown (top right)
2. **View Image**: The current image and prompt are displayed on the left
3. **Answer Questions**: Toggle yes/no switches for each question on the right
4. **Annotate Artifacts**: Click "Annotate Artifacts" to draw on the image
5. **Save & Next**: Click "Next" to save annotations and move to the next image

## API Endpoints

- `GET /api/annotators` - Get list of annotators
- `GET /api/images?annotator=<name>` - Get all images with annotation status
- `GET /api/image/<model>/<dataset_id>/<prompt_id>` - Get image data and annotations
- `POST /api/save_annotations` - Save VQA annotations
- `POST /api/save_artifact_mask` - Save artifact mask

## Data Format

### Annotations (VQA)
```json
{
  "543_0": "yes",
  "543_1": "no"
}
```

### Artifact Masks
Binary PNG masks saved as `<prompt_id>_<annotator>.png`
