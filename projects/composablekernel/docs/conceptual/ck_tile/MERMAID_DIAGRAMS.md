# Mermaid Diagram Management

This document explains how to manage mermaid diagrams in the CK Tile documentation.

## Overview

All mermaid diagrams in the CK Tile documentation have been converted to SVG files for better rendering compatibility. The original mermaid source code is preserved as commented blocks in the RST files, allowing easy updates when needed.

## Directory Structure

- `docs/conceptual/ck_tile/diagrams/` - Contains all SVG diagram files
- `docs/conceptual/ck_tile/convert_mermaid_to_svg.py` - Initial conversion script (one-time use)
- `docs/conceptual/ck_tile/update_diagrams.py` - Helper script to regenerate diagrams from comments

## Diagram Format in RST Files

Each diagram follows this format:

```rst
.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
   .. mermaid::
   
      graph TB
          A --> B
          B --> C

.. image:: diagrams/diagram_name.svg
   :alt: Diagram
   :align: center
```

The commented mermaid block won't appear in the rendered documentation but serves as the source for regenerating the SVG.

## Updating Diagrams

### When to Update

You need to regenerate SVG files when:
- Modifying the mermaid source in a commented block
- Adding new diagrams
- Updating diagram styling

### How to Update

1. **Edit the commented mermaid source** in the RST file
2. **Run the update script**:
   ```bash
   # Update all diagrams
   python docs/conceptual/ck_tile/update_diagrams.py
   
   # Update diagrams in a specific file
   python docs/conceptual/ck_tile/update_diagrams.py transforms.rst
   
   # Force regenerate all diagrams (even if SVGs exist)
   python docs/conceptual/ck_tile/update_diagrams.py --force
   ```

### Prerequisites

The update script requires [mermaid-cli](https://github.com/mermaid-js/mermaid-cli):

```bash
npm install -g @mermaid-js/mermaid-cli
```

## Adding New Diagrams

To add a new mermaid diagram:

1. **Create the commented block** in your RST file:
   ```rst
   .. 
      Original mermaid diagram (edit here, then run update_diagrams.py)
      
      .. mermaid::
      
         graph TB
             A --> B
   ```

2. **Add the image reference** immediately after:
   ```rst
   .. image:: diagrams/my_new_diagram.svg
      :alt: My New Diagram
      :align: center
   ```

3. **Generate the SVG**:
   ```bash
   python docs/conceptual/ck_tile/update_diagrams.py your_file.rst
   ```

## Current Diagrams

The following RST files contain mermaid diagrams (40 total):

- `adaptors.rst` (2 diagrams)
- `convolution_example.rst` (1 diagram)
- `coordinate_movement.rst` (1 diagram)
- `descriptors.rst` (2 diagrams)
- `encoding_internals.rst` (2 diagrams)
- `lds_index_swapping.rst` (3 diagrams)
- `load_store_traits.rst` (2 diagrams)
- `space_filling_curve.rst` (1 diagram)
- `static_distributed_tensor.rst` (1 diagram)
- `sweep_tile.rst` (4 diagrams)
- `tensor_coordinates.rst` (2 diagrams)
- `thread_mapping.rst` (2 diagrams)
- `tile_window.rst` (5 diagrams)
- `transforms.rst` (12 diagrams)

## Troubleshooting

### SVG not generated

- Check that mermaid-cli is installed: `mmdc --version`
- Verify the mermaid syntax is valid
- Look for error messages in the script output

### Diagram not updating

- Use `--force` flag to regenerate: `python docs/update_diagrams.py --force`
- Check that the image reference matches the generated filename

### Pattern not matching

If the update script can't find your commented diagram:
- Ensure proper indentation (3 spaces for comment block content)
- Verify the `.. mermaid::` directive is commented
- Check that the image reference immediately follows the comment block

## Script Details

### update_diagrams.py

This script:
1. Scans RST files for commented mermaid blocks
2. Extracts the mermaid source code
3. Converts to SVG using `mmdc`
4. Saves to the diagrams directory

**Usage:**
- `python docs/conceptual/ck_tile/update_diagrams.py` - Check all files, update missing SVGs
- `python docs/conceptual/ck_tile/update_diagrams.py --force` - Regenerate all SVGs
- `python docs/conceptual/ck_tile/update_diagrams.py <file.rst>` - Update specific file

### convert_mermaid_to_svg.py

This was the initial conversion script. It:
1. Found all active `.. mermaid::` directives
2. Converted them to SVGs
3. Replaced directives with commented source + image references

This script was used once for the initial conversion and typically doesn't need to be run again.
