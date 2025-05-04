# CAD Material Estimator with AI-Powered Counting

This application processes CAD files to identify construction materials, estimate construction costs, and now includes AI-powered counting capabilities using OpenAI's Vision API. It also features SVG-based shape counting for more accurate material quantification.

## Features

- Process DXF, DWG, and IFC files
- Identify construction materials from CAD data
- AI-powered counting using PDF floor plans
- SVG-based shape counting for accurate shape identification and quantification
- Reconcile counts from multiple sources
- Generate detailed cost estimates
- Export results in various formats (JSON, CSV, Excel)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. For SVG-based counting, install additional packages:
   ```bash
   pip install dxf2svg svgelements
   ```

## Configuration

1. Set up your OpenAI API key:
   - Option 1: Set environment variable:
     ```bash
     set OPENAI_API_KEY=your_api_key_here
     ```
   - Option 2: Pass as command line argument:
     ```bash
     python main.py ... --openai-key your_api_key_here
     ```

## Usage

Basic usage with CAD file only:
```bash
python main.py --input path/to/file.dxf --output output_directory
```

Using AI-powered counting with PDF:
```bash
python main.py --input path/to/file.dxf --pdf path/to/floorplan.pdf --output output_directory
```

Using SVG-based shape counting:
```bash
python main.py --input path/to/file.dxf --use-svg-counting --output output_directory
```

Using both SVG-based counting and AI-powered counting:
```bash
python main.py --input path/to/file.dxf --use-svg-counting --pdf path/to/floorplan.pdf --output output_directory
```

### Command Line Arguments

- `--input`, `-i`: Input CAD file path (required)
- `--output`, `-o`: Output directory for results (default: 'output')
- `--location`, `-l`: Project location for cost factors (default: 'Default')
- `--format`, `-f`: Output format (json/csv/xlsx, default: json)
- `--use-llm`: Enable LLM for advanced material identification
- `--pdf`: Path to PDF file for AI-powered counting
- `--openai-key`: OpenAI API key for Vision analysis
- `--use-svg-counting`: Enable SVG-based shape counting for accurate quantification
- `--no-export`: Skip exporting results to files
- `--verbose`, `-v`: Enable verbose output

## Output Files

The program generates several output files in the specified output directory:

- `processed_cad_data.json`: Processed CAD file data
- `materials_data.json`: Identified materials and their properties
- `count.json`: Material counts and categories
- `reconciled_counts.json`: Combined counts from CAD and Vision analysis
- `svg_counts.json`: Shape counts from SVG-based analysis
- Cost estimates in the specified format (JSON/CSV/Excel)

### PDF Processing

When a PDF file is provided:
1. The PDF is converted to high-resolution images
2. Images are analyzed using OpenAI's Vision API
3. The system counts:
   - Doors (all types)
   - Security Cameras/CCTV
   - Furniture (tables, chairs, cabinets, etc.)
4. Counts are reconciled with CAD analysis
5. Final counts are included in the output files

### SVG-Based Shape Counting

When SVG-based counting is enabled:
1. The DWG/DXF file is converted to SVG format
2. SVG elements are analyzed to identify identical shapes
3. Shapes are classified into categories:
   - Doors
   - Windows
   - Furniture
   - Fixtures
   - Security cameras
   - Walls
   - Columns
   - Other
4. Counts are integrated with other material quantities
5. Results are included in the output files

## Error Handling

- If PDF processing fails, the system will continue with CAD analysis only
- If OpenAI API is unavailable, the system will use CAD analysis only
- If SVG conversion fails, the system will use alternative counting methods
- All errors are logged to `cadme.log`

## Dependencies

- ezdxf: CAD file processing
- OpenAI: Vision API for counting
- PyMuPDF: PDF processing
- Pillow: Image processing
- dxf2svg: Converting DXF to SVG for shape analysis
- svgelements: SVG parsing and shape analysis
- Other dependencies listed in requirements.txt

## SVG Shape Counter CLI

A standalone CLI tool is also provided for SVG-based shape counting:

```bash
python svg_counter_cli.py --input path/to/file.dwg --output output_directory
```

This tool converts a DWG/DXF file to SVG and performs shape analysis without the full material estimation process.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 