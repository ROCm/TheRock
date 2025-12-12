# Using PowerPoint Templates with Pandoc

## Quick Answer

```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  --reference-doc=template.pptx
```

---

## Step-by-Step Guide

### Step 1: Create or Get a Template

#### Option A: Use Existing Corporate Template
```bash
# Copy your company's PowerPoint template to docs/development/
cp /path/to/company-template.pptx docs/development/template.pptx
```

#### Option B: Generate Base Template from Pandoc
```bash
cd docs/development

# Generate a basic PowerPoint file first
pandoc STRIX_PRESENTATION.md -o base.pptx

# This creates a default PowerPoint with all necessary layouts
```

### Step 2: Customize the Template (Option B)

1. **Open `base.pptx` in PowerPoint**

2. **Go to View → Slide Master**

3. **Customize Master Slides:**
   - **Title Slide**: First slide layout
   - **Section Header**: For `# Headers`
   - **Title and Content**: For `## Headers`
   - **Two Content**: For two-column layouts
   - **Comparison**: For side-by-side content
   - **Blank**: For custom layouts

4. **Modify Design Elements:**
   - Fonts (Calibri → Your brand font)
   - Colors (Default → Brand colors)
   - Logo (Add company logo to master)
   - Footer (Add copyright/confidential text)
   - Background (Add brand background)

5. **Save as `template.pptx`**

6. **Close Slide Master view**

### Step 3: Use Template with Pandoc

```bash
cd docs/development

pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  --reference-doc=template.pptx
```

---

## Complete Example

### Directory Structure

```
docs/development/
├── STRIX_PRESENTATION.md       # Your content
├── template.pptx                # Your customized template
└── STRIX_PRESENTATION.pptx     # Generated output
```

### Command with All Options

```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  -t pptx \
  --reference-doc=template.pptx \
  --slide-level=2 \
  --toc \
  --toc-depth=1
```

**Options Explained:**
- `-o STRIX_PRESENTATION.pptx` - Output file
- `-t pptx` - Output format (PowerPoint)
- `--reference-doc=template.pptx` - Use custom template
- `--slide-level=2` - `##` becomes new slide (not `#`)
- `--toc` - Generate table of contents slide
- `--toc-depth=1` - Only show top-level headers in TOC

---

## Template Customization Guide

### What Can Be Customized?

#### 1. **Fonts**
```
Slide Master → Fonts dropdown → Create New Theme Fonts
- Heading font
- Body font
```

#### 2. **Colors**
```
Slide Master → Colors dropdown → Create New Theme Colors
- Accent colors (for bullets, highlights)
- Hyperlink colors
- Background colors
```

#### 3. **Layouts**
```
Slide Master → Insert Layout
- Add company-specific layouts
- Custom title slides
- Special section dividers
```

#### 4. **Logo**
```
Slide Master → Insert → Picture
- Place logo on master slide
- Appears on all slides automatically
```

#### 5. **Footer**
```
Slide Master → Insert → Text Box
- Add copyright text
- Add page numbers
- Add confidential notice
```

---

## Advanced Examples

### Multiple Templates

```bash
# Use different templates for different sections
pandoc section1.md -o section1.pptx --reference-doc=template_intro.pptx
pandoc section2.md -o section2.pptx --reference-doc=template_technical.pptx
pandoc section3.md -o section3.pptx --reference-doc=template_conclusion.pptx

# Then manually combine in PowerPoint
```

### Custom Slide Sizes

```bash
# Create template.pptx with custom size in PowerPoint:
# Design → Slide Size → Custom Slide Size (16:9, 4:3, etc.)

# Then use it
pandoc STRIX_PRESENTATION.md \
  -o output.pptx \
  --reference-doc=template_widescreen.pptx
```

### With Syntax Highlighting Theme

```bash
pandoc STRIX_PRESENTATION.md \
  -o output.pptx \
  --reference-doc=template.pptx \
  --highlight-style=tango
```

**Available highlight styles:**
- `pygments` (colorful)
- `tango` (balanced)
- `espresso` (dark)
- `kate` (clean)
- `monochrome` (grayscale)
- `breezedark` (dark theme)

---

## Template Best Practices

### 1. Keep Master Slides Simple

❌ **Don't:**
```
- Too many decorative elements
- Complex backgrounds
- Busy patterns
```

✅ **Do:**
```
- Clean, minimal design
- Subtle backgrounds
- Plenty of white space
```

### 2. Test Template Before Using

```bash
# Create a test markdown file
cat > test.md << 'EOF'
---
title: "Test"
---

# Section 1

## Slide 1

- Bullet 1
- Bullet 2

## Slide 2

```python
def test():
    pass
```

| Column 1 | Column 2 |
|----------|----------|
| A        | B        |
EOF

# Test with template
pandoc test.md -o test_output.pptx --reference-doc=template.pptx

# Check output in PowerPoint
```

### 3. Preserve Required Layouts

Pandoc needs these layouts in the template:
- **Title Slide** (for YAML header)
- **Title and Content** (for `## headers`)
- **Section Header** (for `# headers`)
- **Two Content** (for `::: columns`)
- **Blank** (for custom content)

**Don't delete these!**

---

## Troubleshooting

### Issue: Template Not Applied

**Check:**
```bash
# Ensure template.pptx exists
ls -la template.pptx

# Use absolute path if needed
pandoc STRIX_PRESENTATION.md \
  -o output.pptx \
  --reference-doc=/absolute/path/to/template.pptx
```

### Issue: Fonts Not Appearing

**Solution:**
1. Ensure fonts are installed on your system
2. PowerPoint will substitute if font missing
3. Use web-safe fonts (Arial, Calibri, Times New Roman)

### Issue: Layout Broken

**Solution:**
```bash
# Regenerate base template
pandoc STRIX_PRESENTATION.md -o fresh_base.pptx

# Start customization from scratch
# Don't delete or heavily modify default layouts
```

### Issue: Colors Not Right

**Solution:**
```
1. Open template.pptx
2. View → Slide Master
3. Colors → Customize Colors
4. Set all color slots (not just accents)
5. Save and re-run pandoc
```

---

## Real-World Example: AMD Corporate Template

### Step 1: Prepare Template

```bash
# Assuming you have AMD corporate template
cd docs/development
cp ~/Downloads/AMD_Template.pptx ./amd_template.pptx
```

### Step 2: Verify Template

```bash
# Generate test output
pandoc STRIX_PRESENTATION.md \
  -o test_with_amd_template.pptx \
  --reference-doc=amd_template.pptx

# Open in PowerPoint and verify:
# - Logo appears
# - Fonts are correct
# - Colors match brand
```

### Step 3: Final Generation

```bash
# Generate final presentation
pandoc STRIX_PRESENTATION.md \
  -o STRIX_AI_ML_Presentation_AMD.pptx \
  --reference-doc=amd_template.pptx \
  --slide-level=2
```

---

## Quick Reference

### Basic Command
```bash
pandoc input.md -o output.pptx --reference-doc=template.pptx
```

### With Options
```bash
pandoc input.md \
  -o output.pptx \
  --reference-doc=template.pptx \
  --slide-level=2 \
  --toc \
  --highlight-style=tango
```

### Check Pandoc Version
```bash
pandoc --version
# Ensure version >= 2.0 for best PowerPoint support
```

### List Available Highlight Styles
```bash
pandoc --list-highlight-styles
```

---

## Creating Template from Scratch

### Method 1: PowerPoint First

1. **Create in PowerPoint:**
   ```
   File → New → Blank Presentation
   Design → Slide Size → Widescreen (16:9)
   View → Slide Master
   ```

2. **Customize all layouts**

3. **Add dummy slide with all elements:**
   - Title
   - Bullet points
   - Code block (use monospace font)
   - Table
   - Image placeholder

4. **Save as `template.pptx`**

5. **Delete dummy slide**

6. **Use with pandoc**

### Method 2: Pandoc First (Recommended)

1. **Generate base:**
   ```bash
   pandoc STRIX_PRESENTATION.md -o base.pptx
   ```

2. **Open in PowerPoint**

3. **Modify only master slides:**
   ```
   View → Slide Master
   Modify fonts, colors, add logo
   File → Save As → template.pptx
   ```

4. **Delete all content slides, keep only master**

5. **Use template:**
   ```bash
   pandoc STRIX_PRESENTATION.md \
     -o output.pptx \
     --reference-doc=template.pptx
   ```

---

## Template Repository

### Organize Templates

```bash
docs/development/templates/
├── amd_corporate.pptx          # Main corporate template
├── amd_technical.pptx          # For technical presentations
├── amd_executive.pptx          # For executive briefings
├── amd_widescreen.pptx         # 16:9 format
└── amd_standard.pptx           # 4:3 format
```

### Use Specific Template

```bash
# Executive version
pandoc STRIX_PRESENTATION.md \
  -o executive_brief.pptx \
  --reference-doc=templates/amd_executive.pptx

# Technical version
pandoc STRIX_PRESENTATION.md \
  -o technical_deep_dive.pptx \
  --reference-doc=templates/amd_technical.pptx
```

---

## Summary

**Essential Command:**
```bash
pandoc STRIX_PRESENTATION.md \
  -o output.pptx \
  --reference-doc=template.pptx
```

**Key Points:**
1. Template must be a valid `.pptx` file
2. Keep all default layouts in template
3. Customize via Slide Master view
4. Test with sample content first
5. Use relative or absolute paths

**Result:** Professional presentation with your branding! 🎨

