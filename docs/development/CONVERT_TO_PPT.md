# Converting Strix Documentation to PowerPoint

## Prerequisites

Install pandoc:

```bash
# Ubuntu/Debian
sudo apt-get install pandoc

# macOS
brew install pandoc

# Windows (with Chocolatey)
choco install pandoc

# Or download from: https://pandoc.org/installing.html
```

## Convert to PowerPoint

### Basic Conversion

```bash
cd docs/development

pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  -t pptx
```

### With Custom Theme

```bash
# Use a reference PowerPoint template
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  -t pptx \
  --reference-doc=your_template.pptx
```

### With Syntax Highlighting

```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  -t pptx \
  --highlight-style=tango
```

## File Structure

The presentation markdown (`STRIX_PRESENTATION.md`) is structured as:

- **YAML Header** - Title, author, date
- **Sections** - Separated by `---` (slide breaks)
- **Incremental Lists** - Using `::: incremental`
- **Tables** - Standard markdown tables
- **Code Blocks** - With syntax highlighting

## Customization Tips

### 1. Slide Layouts

Pandoc automatically uses:
- **Title Slide** - First slide with YAML metadata
- **Section Headers** - `# Header` → Section divider slide
- **Content Slides** - `## Header` → Regular content slide

### 2. Incremental Reveals

```markdown
::: incremental
- Point 1 appears first
- Point 2 appears second
- Point 3 appears third
:::
```

### 3. Two-Column Layouts

```markdown
::: columns
:::: column
Left content
::::

:::: column
Right content
::::
:::
```

### 4. Speaker Notes

```markdown
::: notes
These notes will only appear in presenter view
:::
```

## Output Options

### Generate PDF (for sharing)

```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pdf \
  -t beamer
```

### Generate HTML Slides (web-based)

```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.html \
  -t slidy \
  -s --self-contained
```

## Troubleshooting

### Issue: ASCII Art Not Rendering

**Solution:** Use `STRIX_PRESENTATION.md` instead of `STRIX_CLIENT_ARCHITECTURE.md`
- The presentation version uses tables and bullet points
- ASCII art doesn't convert well to PowerPoint

### Issue: Code Blocks Look Bad

**Solution:** Reduce code block size or split across slides

```markdown
## Slide 1: Code Example

```python
# Shorter code snippet
def test():
    pass
```
```

### Issue: Tables Too Wide

**Solution:** Simplify table or split into multiple slides

```markdown
## Part 1: Table Summary

| Key | Value |
|-----|-------|
| A   | 1     |

## Part 2: Table Details

| Key | Details |
|-----|---------|
| A   | More info |
```

### Issue: Images Not Showing

**Solution:** Use relative paths or embed images

```markdown
![Architecture](./images/architecture.png)
```

## Best Practices

1. **Keep it Simple** - One main point per slide
2. **Use Bullets** - Not paragraphs
3. **Limit Code** - Show key snippets only
4. **Use Tables** - For comparison data
5. **Add Breaks** - Use `---` liberally for new slides

## Example: Creating Custom Template

1. Generate initial PowerPoint:
   ```bash
   pandoc STRIX_PRESENTATION.md -o base.pptx
   ```

2. Open `base.pptx` in PowerPoint

3. Modify:
   - Master slide layouts
   - Fonts and colors
   - Company branding

4. Save as `template.pptx`

5. Use template:
   ```bash
   pandoc STRIX_PRESENTATION.md \
     -o STRIX_PRESENTATION.pptx \
     --reference-doc=template.pptx
   ```

## Quick Commands

**Basic conversion:**
```bash
pandoc STRIX_PRESENTATION.md -o output.pptx
```

**With all options:**
```bash
pandoc STRIX_PRESENTATION.md \
  -o STRIX_PRESENTATION.pptx \
  -t pptx \
  --toc \
  --slide-level=2 \
  --highlight-style=tango
```

## Additional Resources

- [Pandoc Manual](https://pandoc.org/MANUAL.html)
- [PowerPoint Output](https://pandoc.org/MANUAL.html#producing-slide-shows-with-pandoc)
- [Markdown Guide](https://www.markdownguide.org/extended-syntax/)

## Files

- `STRIX_PRESENTATION.md` - Pandoc-optimized presentation (use this!)
- `STRIX_CLIENT_ARCHITECTURE.md` - Detailed documentation (not for pandoc)
- `CONVERT_TO_PPT.md` - This file

