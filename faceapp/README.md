# Face Matcher - Image-based Face Recognition

A new face recognition application that takes an input image and finds matching faces in your image database with accuracy scores.

## Features

- **Image-based processing** - No video required, just upload an image
- **Multiple face detection** - Finds all faces in the input image
- **Accuracy scoring** - Shows similarity scores and accuracy percentages
- **Flexible threshold** - Adjustable similarity threshold (0.1-0.9)
- **Two interfaces** - Command line and GUI versions
- **JSON export** - Save results for further processing

## Files

- `face_matcher.py` - Command line version
- `face_matcher_gui.py` - GUI version with visual interface
- `README.md` - This documentation

## Requirements

Make sure you have the virtual environment activated:
```bash
venv310/Scripts/activate
```

## Usage

### Command Line Version

```bash
# Basic usage
python faceapp/face_matcher.py path/to/input/image.jpg

# With custom settings
python faceapp/face_matcher.py input.jpg --threshold 0.3 --images-folder images --output results.json

# Help
python faceapp/face_matcher.py --help
```

**Arguments:**
- `input_image` - Path to the image you want to analyze
- `--images-folder, -d` - Folder containing reference images (default: images)
- `--threshold, -t` - Similarity threshold 0.1-0.9 (default: 0.4)
- `--output, -o` - Save results to JSON file
- `--model, -m` - InsightFace model to use (default: buffalo_l)

### GUI Version

```bash
python faceapp/face_matcher_gui.py
```

**GUI Features:**
- Browse and select input images
- Adjustable threshold slider
- Real-time database status
- Progress bar during processing
- Scrollable results view
- Save results to JSON

## Example Output

```
📊 Processing Results:
   Input Image: test_image.jpg
   Faces Detected: 2
   Database Size: 20 faces
   Processing Time: 1.234s
   Threshold: 0.40
------------------------------------------------------------

👤 Face #1:
   📍 Detection Confidence: 0.95
   🎯 Matches Found: 3

   #1 Match:
      👤 Name: charaf
      📊 Similarity: 0.8234
      🎯 Accuracy: 82.34%
      📁 File: charaf.jpg

   #2 Match:
      👤 Name: charaf2
      📊 Similarity: 0.7123
      🎯 Accuracy: 71.23%
      📁 File: charaf2.jpg

👤 Face #2:
   ❌ No matches found (below threshold)
```

## How It Works

1. **Database Loading** - Loads all faces from the `images/` folder
2. **Face Detection** - Detects all faces in the input image
3. **Embedding Extraction** - Converts faces to numerical embeddings
4. **Similarity Matching** - Compares embeddings using cosine similarity
5. **Threshold Filtering** - Only returns matches above the threshold
6. **Results Ranking** - Sorts matches by similarity score

## Threshold Guide

- **0.1-0.3** - Very lenient, may have false positives
- **0.4-0.6** - Balanced (recommended)
- **0.7-0.9** - Very strict, may miss some matches

## Database Structure

The app expects a folder structure like:
```
images/
├── person1.jpg
├── person2.jpg
├── person3.png
└── ...
```

Each image should contain one clear face. The filename (without extension) becomes the person's name.

## Performance

- **Processing time**: ~1-3 seconds per image
- **Memory usage**: ~500MB-1GB (depends on database size)
- **Accuracy**: High with good quality images
- **GPU acceleration**: Automatic if CUDA is available

## Troubleshooting

**No faces detected:**
- Check image quality and lighting
- Ensure face is clearly visible
- Try different threshold values

**No matches found:**
- Lower the threshold
- Add more reference images
- Check image quality

**Slow performance:**
- Reduce database size
- Use smaller images
- Check GPU availability

## Integration

The results are returned in JSON format, making it easy to integrate with other applications:

```json
{
  "success": true,
  "input_image": "test.jpg",
  "faces_detected": 1,
  "processing_time": 1.234,
  "database_size": 20,
  "threshold_used": 0.4,
  "results": [
    {
      "face_index": 0,
      "bbox": [100, 150, 200, 250],
      "detection_confidence": 0.95,
      "matches": [
        {
          "name": "charaf",
          "similarity_score": 0.8234,
          "accuracy_percentage": 82.34,
          "matched_image_path": "images/charaf.jpg",
          "matched_filename": "charaf.jpg"
        }
      ]
    }
  ]
}
``` 