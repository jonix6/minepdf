
from functools import reduce
import math
from io import BytesIO

import re

from PIL import Image
from pdfminer.utils import apply_matrix_pt


LINE_END_MARKS = r'。？！.!?…'
SUFFIXES = r'）》』】)>]}”"'
LINE_ENDS = rf'[{LINE_END_MARKS}][{SUFFIXES}]*$'


def matrix_transform(pt, matrix):
    a, b, c, d, e, f = matrix
    x, y = pt
    return a*x+c*y+e, b*x+d*y+f


def distance(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def rect(bbox, vertical=False):
    xmin = min(map(lambda p: p[0], bbox))
    xmax = max(map(lambda p: p[0], bbox))
    ymin = min(map(lambda p: p[1], bbox))
    ymax = max(map(lambda p: p[1], bbox))
    if vertical:
        return xmax, ymin, xmin, ymax
    return xmin, ymin, xmax, ymax


class TextAnalyzer:
    def __init__(self, **kwargs):
        self.fontsize_between_ratio = kwargs.get('fontsize_between_ratio', 0.9)
        self.fontsize_linebreak_ratio = kwargs.get(
            'fontsize_linebreak_ratio', 0.95)
        self.max_line_spacing = kwargs.get('max_line_spacing', 0.6)
        self.max_word_spacing = kwargs.get('max_word_spacing', 1.5)
        self.indent_length = kwargs.get('indent_length', 2.3)
        self.min_column_gap = kwargs.get('min_column_gap', 0.75)
        self.space_width = kwargs.get('space_width', 0.25)
        self.check_layer = kwargs.get('check_layer', False)

    def is_connected(self, line1, line2):
        fsz = line1.get_fontsize() / line2.get_fontsize()
        "average fontsize on both lines varies, that means they are with different styles."
        proof1 = round(min(fsz, 1 / fsz), 1) < self.fontsize_between_ratio
        if proof1:
            fsz = line1.children[-1].fontsize / line2.children[0].fontsize
            proof1 = round(min(fsz, 1 / fsz),
                           1) < self.fontsize_linebreak_ratio

        prevText = line1.children[-1].text
        proof2 = True
        if prevText:
            "if prev line ends with common ending character, it could make a line break."
            proof2 = bool(re.match(LINE_ENDS, prevText))
        return not (proof1 or proof2)

    def group_lines(self, spans):
        groups = []
        for span in spans:
            inline = False
            for line in groups:
                index = line.inline_with(
                    span, self.max_line_spacing, self.check_layer)
                if index is not None:
                    line.add(span, index)
                    inline = True
                    break
            if not inline:
                groups.append(textLineBox(span, self.space_width))

        lines = []
        for line in groups:
            for new_line in self.split_line(line):
                lines.append(new_line)
        return lines

    def split_line(self, line):
        spans = len(line.children)
        new_line = None
        for i in range(spans):
            if i == spans - 1:
                break
            box1, box2 = line.children[i:i+2]
            if not new_line:
                new_line = textLineBox(box1, self.space_width)
            _, __, ax1, ay1 = rect(box1.bbox)
            bx0, by0, _, __ = rect(box2.bbox)
            maxDist = min(box1.fontsize, box2.fontsize) * self.max_word_spacing
            dist = (by0 - ay1) if line.vertical else (bx0 - ax1)
            if dist < maxDist:
                new_line.add(box2)
            else:
                yield new_line
                new_line = textLineBox(box2, self.space_width)
        if not new_line:
            yield line
        else:
            yield new_line

    def group_by_aligns(self, lines, groups=None):
        groups = groups or []
        for line in lines:
            if line.is_empty():
                continue
            ingroup = False
            for para in reversed(groups):
                index = para.align_with(line, indent=self.indent_length)
                if index is not None:
                    para.add(line, index)
                    ingroup = True
                    break
            if not ingroup:
                groups.append(textParaBox(line))

        split_lines = []
        real_groups = []
        for para in groups:
            columns = para.get_columns(self.min_column_gap)
            if not columns:
                real_groups.append(para)
                continue
            split_lines += list(para.split(columns=columns))

        if split_lines:
            real_groups = self.group_by_aligns(
                split_lines[:], groups=real_groups[:])

        return real_groups

    def group_paras(self, lines):
        paras = []
        for para in self.group_by_aligns(lines):
            for new_para in self.split_para(para):
                paras.append(new_para)
        return paras

    def split_para(self, para):
        lines = len(para.children)
        new_para = None
        i = 0
        while i < lines - 1:
            prev, cur = para.children[i:i+2]
            if not new_para:
                new_para = textParaBox(prev)
            prev_ind = new_para.indent[-1]

            al, at, ar, ab = rect(prev.bbox)
            bl, bt, br, bb = rect(cur.bbox)

            if para.vertical:
                al, at, ar, ab = at, ar, ab, al
                bl, bt, br, bb = bt, br, bb, bl

            vDist = bt - ab
            maxVDist = min(prev.get_fontsize(), cur.get_fontsize()
                           ) * self.max_line_spacing

            if vDist >= maxVDist:
                yield new_para
                new_para = textParaBox(cur)
                i += 1
                continue

            fontsize = min(prev.get_fontsize(), cur.get_fontsize())
            maxHDist = fontsize * self.indent_length

            cur_indent = bl - al

            if abs(cur_indent) > maxHDist:
                """
                =========|		      =====|
                    =====|		===========|
                indent too long, split.
                but, what if it is a floating paragraph?
                """
                yield new_para
                new_para = textParaBox(cur)
                i += 1
                continue

            connected = self.is_connected(prev, cur)

            right_indent = br - ar
            if abs(cur_indent) < fontsize:
                """
                ========|
                ========|
                both lines left aligned, determine if it's a line wrap.
                """
                if right_indent > fontsize:
                    """
                    ==========
                    ============
                    this should be a new paragraph.
                    """
                    yield new_para
                    new_para = textParaBox(cur)
                    i += 1
                    continue
                else:
                    """
                    ==========		=========
                    ==========		=======
                    seems like a line wrap, but if it's a short paragraph?
                    """
                    if not connected:
                        "prove they are connected, in semantic and graphic way."
                        yield new_para
                        new_para = textParaBox(cur)
                        i += 1
                        continue
            else:
                """
                ========|		  ======|
                ======|		========|
                both lines not left aligned, determine if it's a line wrap on an indented paragraph.
                """
                if right_indent > fontsize:
                    """
                    ==========		  =========
                    ==========	=============
                    this should be a new paragraph.
                    """
                    yield new_para
                    new_para = textParaBox(cur)
                    i += 1
                    continue
                elif abs(prev_ind - cur_indent) < fontsize:
                    """
                    ========		=========
                    ==========		  =======
                    =====			======
                    current line is left aligned with current indent, which means it's a new paragraph.
                    """
                    yield new_para
                    new_para = textParaBox(cur)
                    i += 1
                    continue
                else:
                    """
                    ========		  ========
                    ========		  ========
                    =====			=======
                    that seems the first line wrap, prove it like above.
                    """
                    if not connected:
                        yield new_para
                        new_para = textParaBox(cur)
                        i += 1
                        continue
                    "now something happens: previous line is the first line of a paragraph. If current paragraph has multiple lines, it should be splitted."
                    if len(new_para.children) > 1:
                        children = new_para.children[:]
                        new_para = textParaBox(children[0])
                        for j in range(1, len(children) - 1):
                            new_para.add(children[j])
                        yield new_para
                        new_para = textParaBox(prev)

            new_para.add(cur)
            i += 1

        if not new_para:
            yield para
        else:
            yield new_para


class textSpanBox:
    # unique style, unable line wrapping
    def __init__(self, point, seq, style, layerName='', matrix=[1, 0, 0, 1, 0, 0]):
        self.style = style
        self.font = self.style.font
        self.fontsize = self.style.fontsize
        self.scaling = self.style.scaling * 0.01
        self.layerName = layerName

        self.matrix = matrix
        a, b, c, d, *_ = matrix

        self.upright = 0 < a*d*self.style.scaling and b*c <= 0
        self.vertical = bool(
            self.font.vertical - abs(0 - all([b, c]) and not all([a, d])))

        self.text = ''

        self.seq = seq

        x0, y0 = matrix_transform(point, matrix)
        # visual metrics and original metrics
        width = height = ind = ow = oh = oind = 0
        if self.vertical:
            width, height, ow, oh = height, width, oh, ow
        for obj in seq:
            if isinstance(obj, (int, float)):  # spacing or indenting
                x = obj * 0.001 * self.style.fontsize * self.scaling
                if not self.text:
                    oind -= x
                else:
                    ow -= x
                p0 = matrix_transform((0, 0), self.matrix)
                p1 = matrix_transform((x, 0), self.matrix)
                x = math.copysign(distance(p0, p1), x)
                if not self.text:
                    ind -= x
                else:
                    width -= x
                continue
            for cid in self.font.decode(obj):
                char, cwidth, _, cheight = self.get_char(
                    cid, addcharspace=True)
                if not cwidth:
                    continue
                if char.isspace() and not self.text:
                    oind += cwidth
                    ind += distance(
                        matrix_transform((cwidth, 0), matrix),
                        matrix_transform((0, 0), matrix))
                    continue

                ow += cwidth
                oh = max(oh, cheight)
                tl, tr, br, bl = map(lambda p: matrix_transform(p, matrix), [
                    (0, 0), (cwidth, 0), (cwidth, cheight), (0, cheight)
                ])
                width += distance(tl, tr)
                height = max(height, max(distance(tl, bl), distance(tr, br)))
                self.text += char

        # line height
        ascent = self.font.ascent / (self.font.get_height() or 1)
        x, y = point
        x += oind
        self.originbox = [(x, y-oh*ascent), (x+ow, y-oh*ascent),
                          (x+ow, y), (x, y)]
        x0 += ind
        self.bbox = [(x0, y0 - height*ascent), (x0 + width, y0 - height*ascent),
                     (x0 + width, y0), (x0, y0)]

        x0, y0 = matrix_transform((0, 0), matrix)
        x1, y1 = matrix_transform((self.fontsize, 0), matrix)
        self.fontsize = distance((x0, y0), (x1, y1))

    def get_char(self, cid, addcharspace=False):
        width, lsb = self.font.metrics[cid]
        char = self.font.to_unicode(cid)
        width *= self.style.fontsize * self.scaling
        lsb *= self.style.fontsize * self.scaling
        if addcharspace:
            width += self.style.charspace
        if char.isspace():
            width += self.style.wordspace

        height = self.font.get_height() * self.style.fontsize * \
            self.scaling - self.style.rise
        return char, width, lsb, height


class textLineBox:
    def __init__(self, spanbox, space_width=0.25):
        self.children = [spanbox]
        self.vertical = spanbox.vertical
        self.layerName = spanbox.layerName
        self.bbox = spanbox.bbox
        self.tabs = []
        self.space_width = space_width

    def __getitem__(self, key):
        return self.children[key]

    def inline_with(self, spanbox, maxVDist, check_layer=True):
        if (spanbox.vertical != self.vertical) or (check_layer and self.layerName != spanbox.layerName):
            return None

        ax0, ay0, _, ay1 = rect(spanbox.bbox, self.vertical)
        _, by0, __, by1 = rect(self.bbox, self.vertical)
        if abs(ay0 - by0) > max(ay1 - ay0, by1 - by0) * maxVDist:
            return None
        return next((
            i for i, child in enumerate(self.children)
            if ax0 < rect(child.bbox, self.vertical)[0]), len(self.children))

    def add(self, spanbox, index=-1):
        if index < 0:
            index = len(self.children)
        self.children.insert(index, spanbox)

        cur_box = rect(spanbox.bbox, self.vertical)
        prev_box = next_box = None
        if index < len(self.children) - 1:
            next_box = rect(self.children[index+1].bbox, self.vertical)
        if index > 0:
            prev_box = rect(self.children[index-1].bbox, self.vertical)

        if next_box is not None:
            self.tabs.insert(index, (cur_box[2], next_box[0]))
        if prev_box is not None:
            if index >= len(self.children) - 1:
                self.tabs.insert(index-1, (prev_box[2], cur_box[0]))
            else:
                self.tabs[index-1] = (prev_box[2], cur_box[0])

        self.expand(spanbox)

    def expand(self, box):
        ax0, ay0, ax1, ay1 = rect(box.bbox, self.vertical)
        bx0, by0, bx1, by1 = rect(self.bbox, self.vertical)

        xmin = min(ax0, bx0)
        xmax = max(ax1, bx1)
        ymin = min(ay0, by0)
        ymax = max(ay1, by1)
        self.bbox = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def get_text(self, sep='', space_width=0.25):
        text = ''
        fsz = self.get_fontsize()
        if fsz:
            for i, span in enumerate(self.children):
                prefix = ''
                tab = 0
                if not i == 0:
                    tab = self.tabs[i-1][1] - self.tabs[i-1][0]
                    prefix = sep
                text += prefix
                if tab > fsz * self.space_width:
                    text += ' ' * math.floor(tab / fsz / self.space_width)
                text += span.text
        return text

    def get_fontsize(self):
        totalchars = 0
        totalsize = 0
        for span in self.children:
            if not span.text or span.text.isspace():
                continue
            numchars = 0
            for c in span.text:
                if c.isspace():
                    continue
                numchars += 1
            totalsize += span.fontsize * numchars
            totalchars += numchars
        if not totalchars:
            return 0
        return totalsize / totalchars

    def is_empty(self):
        return not self.get_text().strip()

    def is_unique(self):
        fonts = set()
        max_fontsize = 0
        for span in self.children:
            max_fontsize = max(max_fontsize, span.fontsize)
            font_scale = max_fontsize / span.fontsize
            if font_scale > 1.05 or font_scale < 0.95:
                return False
            fonts.add(span.font.fontname)
        return len(fonts) <= 1


class textParaBox:
    def __init__(self, linebox):
        self.children = [linebox]
        self.vertical = linebox.vertical
        self.bbox = linebox.bbox
        self.indent = [0]

    def __getitem__(self, key):
        return self.children[key]

    def align_with(self, linebox, indent=2.2):
        if linebox.vertical != self.vertical:
            return None

        cur_box = rect(linebox.bbox, vertical=linebox.vertical)

        size = linebox.get_fontsize()
        prev_l = None
        i = 0
        while i < len(self.children):
            next_l = self.children[i]
            next_scale = next_l.get_fontsize() / size
            if not 0.55 < next_scale < 1.95:
                return None

            next_box = rect(next_l.bbox, vertical=self.vertical)
            if cur_box[1] >= next_box[1]:
                prev_l = next_l
                i += 1
                continue
            else:
                break

        next_dist = abs(next_box[0] - cur_box[0])
        next_indent = size * max(1, next_scale) * indent
        proof1 = next_dist <= next_indent
        proof2 = False
        if prev_l is not None:
            prev_box = rect(prev_l.bbox, vertical=self.vertical)
            prev_dist = abs(prev_box[0] - cur_box[0])
            prev_indent = max(size, prev_l.get_fontsize()) * indent
            proof2 = prev_dist <= prev_indent

        if proof1 or proof2:
            return i
        return None

    def add(self, linebox, index=-1):
        ax0, ay0, *_ = rect(linebox.bbox)
        firstLine = self.children[0]
        bbox = rect(firstLine.bbox)
        firstLeft = bbox[1] if self.vertical else bbox[0]
        boxLeft = ay0 if self.vertical else ax0

        if index < 0:
            index = len(self.children)
        self.children.insert(index, linebox)
        self.indent.insert(index, firstLeft - boxLeft)
        self.expand(linebox)

    def expand(self, box):
        ax0, ay0, ax1, ay1 = rect(box.bbox)
        bx0, by0, bx1, by1 = rect(self.bbox)

        xmin = min(ax0, bx0)
        xmax = max(ax1, bx1)
        ymin = min(ay0, by0)
        ymax = max(ay1, by1)
        self.bbox = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    def get_text(self, sep='', linesep=''):
        text = ''
        for line in self.children:
            text += line.get_text(sep=sep) + linesep
        return text

    def get_columns(self, minColumnGapEm=0.5):
        unique_tabs = []
        for r, line in enumerate(self.children):
            if not line.tabs:
                continue
            minsize = line.get_fontsize() * minColumnGapEm
            tabs = []
            for i, tab in enumerate(line.tabs):
                if tab[1] - tab[0] < minsize:
                    continue
                tabs.append(([(r, i)], tab))

            if not unique_tabs:
                unique_tabs = tabs
                continue

            for ([i], tab) in tabs:
                for j, (indexes, _tab) in enumerate(unique_tabs):
                    proof = abs(tab[1] - _tab[1]) < minsize
                    """
					=========  =======
					=========  =======
					=========  =======
					"""
                    if proof and indexes[-1][0] + 1 == i[0]:
                        indexes.append(i)
                        unique_tabs[j] = indexes, _tab

        result = []
        for (indexes, tab) in unique_tabs:
            if len(indexes) <= 1:
                continue
            result.append(indexes)
        return result

    def split(self, columns=[]):
        columns = reduce(lambda x, y: x + y, columns)
        row_indexes = []
        for (r, c) in columns:
            if not r in row_indexes:
                row_indexes.append(r)

        for r, line in enumerate(self.children):
            if not r in row_indexes:
                yield line
                continue
            new_line = None
            c = 0
            while c < len(line.children) - 1:
                span1, span2 = line.children[c:c+2]
                if not new_line:
                    new_line = textLineBox(span1)
                if not (r, c) in columns:
                    new_line.add(span2)
                else:
                    yield new_line
                    new_line = textLineBox(span2)
                c += 1
            yield new_line
