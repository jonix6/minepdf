
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from converter import Device, Interpreter


def page_range_match(s, maxsize=10000):
    def get_range(x):
        if x.isdigit():
            return {min(int(x), maxsize)}
        start, _, end = x.partition('-')
        if start.strip().isdigit() and end.strip().isdigit():
            return set(range(min(int(start.strip()), maxsize), min(int(end.strip()) + 1, maxsize)))
        return set()
    result = set()
    for x in s.strip().split(','):
        result |= get_range(x.strip())
    return result


def render_pages(ipt, page_range='*', laparams=None, filtered=[], password=b'', check_visible=True, maxsize=10000):
    page_range = page_range.strip()
    parser = PDFParser(ipt)
    doc = PDFDocument(parser, password=password, fallback=False)

    renderer = Device(filtered=filtered, laparams=laparams,
                      check_visible=check_visible)
    factory = Interpreter(renderer)

    wildcard = page_range == '*'
    pages = not wildcard and page_range_match(page_range)
    endpage = max(pages or [0])
    if wildcard or endpage:
        for i, page in enumerate(PDFPage.create_pages(doc)):
            i += 1
            if not wildcard and i not in pages:
                continue
            factory.process_page(page)
            yield i, renderer, factory
            if not wildcard and i == endpage:
                break
    factory.close()


text_sort_key = {
    'left': (lambda p: p.bbox[0]),
    'center': (lambda p: (p.bbox[0][1], p.bbox[0][0])),
    'right': (lambda p: (p.bbox[1][1], p.bbox[1][0]))
}


def render_text(ipt, page_range='*', laparams=None, box_align='center', password=b'', check_visible=True, maxsize=10000):
    for pageno, renderer, _ in render_pages(
            ipt, page_range=page_range, laparams=laparams,
            filtered=['image', 'path'], password=password,
            check_visible=check_visible, maxsize=maxsize):
        texts = []
        for para in sorted(renderer.text_layer, key=text_sort_key[box_align]):
            texts.append([line.get_text() for line in para])
        yield pageno, texts


def render_image(ipt, page_range='*', password=b'', maxsize=10000):
    for pageno, renderer, _ in render_pages(
            ipt, filtered=['text', 'path'], page_range=page_range,
            password=password, maxsize=maxsize):
        for objid, (name, stream, bbox, matrix) in renderer.images.items():
            yield pageno, objid, name, stream, bbox, matrix

