# MinePDF

MinePDF is a pure-Python tool for extracting reflowable texts, images, fonts and other contents from PDF documents. This project is written based on [PDFMiner](https://github.com/pdfminer/pdfminer.six), and performs a lot of improvements to make it easy for analyzing visual contents on pages.

This project is still under developing. Documentation, test suite and debugging will be completed later.

## Improved features

+ Variable parameters for grouping text lines into paragraphs
+ Better image extraction using Pillow and Numpy
+ Various color spaces support for color conversion
+ Improved font parsing for glyph mapping and font extraction
+ Improved render device for checking if objects visible
+ Customizable extraction: page range selection and resource type filter
+ Customizable logging system

## Basic workflow

```python
from minepdf import render_pages

fp = open('sample.pdf', 'rb')
pages = render_pages(fp, page_range='1-5,20-100', filtered=['path'])
for pageno, renderer, factory in pages: # extract each page
    handle_text(renderer.text_layer) # handle paragraphs
    handle_image(renderer.images) # handle images
    handle_resources(factory) # handle other resources (fonts, curves, etc.)
fp.close()
```

## Text layout analysis

According to the LAParams in PDFMiner for layout analysis, this project provides multiple parameters for text layout analysis, even grouping text fragments into text lines and paragraphs.

| Parameter                  | Definition                                                   |
| -------------------------- | ------------------------------------------------------------ |
| `fontsize_between_ratio`   | minimum average font size ratio between both lines           |
| `fontsize_linebreak_ratio` | minimum average font size ratio between the last character of the upper text line and the first character of the lower text line |
| `max_line_spacing`         | maximum spacing between both text lines                      |
| `max_word_spacing`         | maximum spacing between words in one line                    |
| `indent_length`            | length of the indentation                                    |
| `min_column_gap`           | minimum column gap length among multi-column text blocks     |
