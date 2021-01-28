
from pdfminer.psparser import literal_name, LIT, PSLiteral
from pdfminer.pdftypes import list_value, dict_value, stream_value, int_value, num_value, resolve1
from pdfminer.latin_enc import ENCODING
from pdfminer.cmapdb import CMapDB, CMapParser, FileUnicodeMap, IdentityCMap
import struct

from fontTools import ttLib, cffLib
from fontTools.misc import psLib, eexec
from fontTools.misc.psCharStrings import T1CharString
from io import BytesIO
from collections import defaultdict
import cidsystem
from itertools import chain


class FontFile:
    def __init__(self, obj):
        self.fp = BytesIO(stream_value(obj).get_data())
        self.cid = False
        self.font = None

    def get_charset(self):
        # charset index -> glyph name
        return []

    def get_glyphset(self):
        # charset index -> glyph
        return []

    def get_charmap(self):
        # charset index -> unicode
        return {}

    def get_metrics(self):
        # charset index -> left, top, right, bottom
        return []

    def close(self):
        self.fp.close()


class T1FontFile(FontFile):
    def __init__(self, obj):
        super().__init__(obj)
        self.hscale = self.vscale = 1000
        self.parse()

        self.glyphset = defaultdict()
        self.metrics = defaultdict()

    def parse(self):
        self.font = psLib.suckfont(self.fp.read())
        charStrings = self.font["CharStrings"]
        del charStrings['.notdef']

        lenIV = self.font["Private"].get("lenIV", 4)
        assert lenIV >= 0
        subrs = self.font["Private"].setdefault("Subrs", [])
        for glyphName, charString in charStrings.items():
            charString = eexec.decrypt(charString, 4330)[0]
            charStrings[glyphName] = T1CharString(
                charString[lenIV:], subrs=subrs)
        for i in range(len(subrs)):
            charString = eexec.decrypt(subrs[i], 4330)[0]
            subrs[i] = T1CharString(charString[lenIV:], subrs=subrs)

    def keys(self):
        return self.font.keys()

    def get_charset(self):
        return self.font['Encoding']

    def get_glyphset(self):
        if not self.glyphset:
            for uid, glyph in self.font['CharStrings'].items():
                if self.cid:
                    uid = self.font['Encoding'].index(uid)
                self.glyphset[uid] = glyph
        return self.glyphset

    def get_metrics(self):
        if not self.metrics:
            charset = self.get_charset()
            for gid, glyph in self.font['CharStrings'].items():
                if gid in charset:
                    cid = charset.index(gid)
                    bounds = glyph.calcBounds(self.font['CharStrings'])
                    if bounds:
                        self.metrics[cid] = (
                            (bounds[2] - bounds[0]) / self.hscale, 0)
        return self.metrics


class TTFontFile(FontFile):
    def __init__(self, obj):
        super().__init__(obj)
        self.font = ttLib.TTFont(self.fp, fontNumber=0)

        self.numGlyphs = int(self.font['maxp'].numGlyphs)
        self.hscale = self.vscale = self.font['head'].unitsPerEm
        if self.font.has_key('OS/2'):
            self.ascent = self.font['OS/2'].sTypoAscender / self.vscale
            self.descent = self.font['OS/2'].sTypoDescender / self.vscale
        elif self.font.has_key('hhea'):
            self.ascent = self.font['hhea'].ascent / self.vscale
            self.descent = self.font['hhea'].descent / self.vscale

        self.has_cmap = self.font.has_key('cmap')
        self.charmap = defaultdict()
        self.glyphset = defaultdict()
        self.metrics = defaultdict()

    def keys(self):
        return self.font.keys()

    def get_charset(self):
        return self.has_cmap and self.font.getGlyphOrder()

    def get_charmap(self):
        if self.has_cmap and not self.charmap:
            charset = self.get_charset() or []

            for table in self.font['cmap'].tables:
                if table.isUnicode():
                    items = chain(*table.uvsDict.values()
                                  ) if table.format == 14 else table.cmap.items()
                    for (ucode, gid) in items:
                        if ucode:
                            if not self.cid:
                                self.charmap[gid] = chr(ucode)
                            elif gid in charset:
                                self.charmap[charset.index(gid)] = chr(ucode)
        return self.charmap

    def get_glyphset(self):
        if not self.glyphset:
            data = self.font.reader['glyf']
            Glyph = ttLib.getTableModule('glyf').Glyph
            loca = self.font['loca']
            last = int(loca[0])
            charset = self.get_charset()

            for i in range(1, len(loca)):
                next = int(loca[i])
                glyphdata = data[last:next]
                if glyphdata:
                    if not self.cid:
                        i = charset[i]
                    self.glyphset[i] = Glyph(glyphdata)
                    last = next
        return self.glyphset

    def get_metrics(self):
        if not self.metrics:
            data = self.font.reader['hmtx']
            numberOfMetrics = min(
                int(self.font['hhea'].numberOfHMetrics), self.numGlyphs)

            metricsFmt = '>' + 'Hh' * numberOfMetrics
            _metrics = struct.unpack(metricsFmt, data[:numberOfMetrics << 2])
            data = data[numberOfMetrics << 2:]
            numberOfSideBearings = self.numGlyphs - numberOfMetrics
            sideBearings = struct.unpack(
                'h' * numberOfSideBearings, data[:numberOfSideBearings << 1])

            for i in range(numberOfMetrics):
                advanceWidth, lsb = _metrics[i << 1:i+1 << 1]
                self.metrics[i] = (
                    advanceWidth / self.hscale, lsb / self.hscale)
            lastAdvance = _metrics[-2]
            for i in range(numberOfSideBearings):
                self.metrics[i + numberOfMetrics] = (lastAdvance / self.hscale,
                                                     sideBearings[i] / self.hscale)
        return self.metrics


class CFF_FontFile(FontFile):
    def __init__(self, obj):
        super().__init__(obj)
        fontset = cffLib.CFFFontSet()
        isCFF2 = struct.unpack('3B', self.fp.read(3))[0] == 2
        self.fp.seek(0)
        fontset.decompile(self.fp, None, isCFF2=isCFF2)
        self.font = fontset[0]
        self.hscale = self.vscale = 1000
        self.numGlyphs = self.font.numGlyphs

        self.charset = []
        self.glyphset = defaultdict()
        self.metrics = defaultdict()

    def keys(self):
        return list(self.font.rawDict.keys())

    def get_charset(self):
        if not self.charset:
            charStringsAreIndexed = self.font.CharStrings.charStringsAreIndexed
            if not charStringsAreIndexed:
                self.charset = self.font.charset
            else:
                charStrings = self.font.CharStrings.charStrings
                self.charset = ['.notdef'] * self.font.numGlyphs
                for gid, cid in charStrings.items():
                    self.charset[cid] = gid
        return self.charset

    def get_glyphset(self):
        if not self.glyphset:
            for i, glyph in enumerate(self.font.CharStrings.values()):
                if not self.cid:
                    i = self.get_charset()[i]
                self.glyphset[i] = glyph
        return self.glyphset

    def get_metrics(self):
        if not self.metrics:
            for i, charString in enumerate(self.get_glyphset().values()):
                bounds = charString.calcBounds(self.font.CharStrings)
                if bounds:
                    self.metrics[i] = (
                        (bounds[2] - bounds[0]) / self.hscale, 0)
        return self.metrics


class Font:
    max_ascent = [1.9, -1.9]

    def __init__(self, spec, embedFont=None, opentype=False, cid=False):
        self.basefont = literal_name(spec.get('BaseFont', 'unknown'))
        self.opentype = opentype
        self.hscale = self.vscale = 1000

        self.embedFont = None
        if embedFont:
            if hasattr(embedFont, 'fp'):
                self.embedFont = embedFont
            else:
                self.embedFont = self.readEmbedFont(embedFont)
            self.embedFont.cid = cid
            if isinstance(self.embedFont, FontFile) and not cid:
                self.hscale = self.embedFont.hscale
                self.vscale = self.embedFont.vscale

        self.readDescriptor(spec)

        self.charset = []
        self.matrix = list_value(spec.get('FontMatrix', [1, 0, 0, 1, 0, 0]))
        self.vertical = False

        self.custom_map = []

        self.charmap = {}
        self.metrics = defaultdict(lambda: (self.missingWidth, 0))

    def readDescriptor(self, spec):
        descriptor = dict_value(spec.get('FontDescriptor', {}))
        self.flags = int_value(descriptor.get('Flags', 0))
        self.fontname = resolve1(descriptor.get('FontName', 'unknown'))
        self.missingWidth = num_value(descriptor.get(
            'MissingWidth', self.hscale)) / self.hscale
        ascent = num_value(descriptor.get(
            'Ascent', self.vscale * 0.75)) / self.vscale
        capHeight = num_value(descriptor.get(
            'CapHeight', self.vscale * 0.75)) / self.vscale
        ascent = min(self.max_ascent[0], max(abs(ascent), abs(capHeight)))
        self.ascent = ascent
        descent = num_value(descriptor.get(
            'Descent', self.vscale * -0.25)) / self.vscale
        descent = max(self.max_ascent[1], -abs(descent))
        self.descent = descent
        self.italic_angle = num_value(descriptor.get('ItalicAngle', 0))
        self.leading = num_value(descriptor.get('Leading', 0))
        bbox = list_value(descriptor.get(
            'FontBBox', [0, self.descent, self.missingWidth, self.ascent]))
        if any(map(lambda x: x > 1, bbox)):
            bbox = list(map(lambda x, y: x / y, bbox,
                            [self.hscale, self.vscale] * 2))
        self.ascent = max(self.ascent, bbox[3])
        self.descent = min(self.descent, bbox[1])
        self.bbox = bbox

    def readEmbedFont(self, obj):
        stream = stream_value(obj)
        return stream

    def char_metrics(self, cid):
        assert isinstance(cid, int)
        return self.metrics[cid]

    def get_height(self):
        h = self.bbox[3] - self.bbox[1]
        return h or self.ascent - self.descent


class SimpleFont(Font):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype, cid=False)
        self.readSpec(spec)

    def readSpec(self, spec):
        self.glyphmap = ['.notdef'] * 256
        self.charmap = {}
        self.metrics = defaultdict(lambda: (self.missingWidth, 0))

        self.readCharset(spec)
        self.readCharmap(spec)
        self.readMetrics(spec)

    def readCharset(self, spec):
        enc = ['StandardEncoding', 'MacRomanEncoding',
               'WinAnsiEncoding', 'PDFDocEncoding']
        name = 'StandardEncoding'
        diff = []
        if 'Encoding' in spec:
            encoding = resolve1(spec['Encoding'])
            if isinstance(encoding, dict):
                name = literal_name(encoding.get(
                    'BaseEncoding', 'StandardEncoding'))
                diff = list_value(encoding.get('Differences'))
            else:
                name = literal_name(encoding)

        for tpl in ENCODING:
            gid = tpl[0]
            i = enc.index(name) + 1
            order = tpl[i]
            if order is not None:
                self.glyphmap[order] = gid

        cid = 0
        for x in diff:
            if isinstance(x, int):
                cid = x
            elif isinstance(x, PSLiteral):
                self.custom_map.append(cid)
                self.glyphmap[cid] = x.name

    def readCharmap(self, spec):
        if self.embedFont:
            charmap = self.embedFont.get_charmap()
            for gid, uchar in charmap.items():
                if gid in self.glyphmap:
                    self.charmap[self.glyphmap.index(gid)] = uchar

        if 'ToUnicode' in spec:
            strm = stream_value(spec['ToUnicode'])
            charmap = FileUnicodeMap()
            CMapParser(charmap, BytesIO(strm.get_data())).run()
            self.charmap.update(charmap.cid2unichr)

    def readMetrics(self, spec):
        if self.embedFont:
            metrics = self.embedFont.get_metrics()
            for cid, mtx in metrics.items():
                if cid < 256:
                    self.metrics[cid] = mtx

        if 'Widths' in spec:
            firstChar = max(0, min(255, int_value(spec.get('FirstChar', 0))))
            for i, width in enumerate(list_value(spec['Widths'])):
                cid = i + firstChar
                mtx = self.metrics[cid]
                self.metrics[cid] = (width * 0.001, mtx and mtx[1] or 0)

    def decode(self, bytes):
        return bytes

    def to_unicode(self, cid):
        if not isinstance(cid, int):
            cid = ord(cid.name)
        char = self.charmap.get(cid)
        if char is None and not cid in self.custom_map:
            char = chr(cid)
        return char or '\ufffd'


class CIDFont(Font):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype, cid=True)
        self.readSpec(spec)

    def readSpec(self, spec):
        self.charset = IdentityCMap(WMode=0)
        self.charmap = {}
        self.metrics = defaultdict(lambda: (self.missingWidth, 0))

        self.readCharset(spec)
        self.readCharmap(spec)
        self.readMetrics(spec)

    def readCharset(self, spec):
        encoding = resolve1(spec['Encoding'])
        self.charset = CMapDB.get_cmap(literal_name(encoding))
        self.vertical = self.charset.is_vertical()

    def readCharmap(self, spec):
        if 'ToUnicode' in spec:
            strm = stream_value(spec['ToUnicode'])
            charmap = FileUnicodeMap()
            CMapParser(charmap, BytesIO(strm.get_data())).run()
            self.charmap = charmap.cid2unichr.copy()

        if 'CIDSystemInfo' in spec:
            charmapInfo = dict_value(spec['CIDSystemInfo'])
            Registry, Ordering = charmapInfo.get(
                'Registry', ''), charmapInfo.get('Ordering', '')
            Registry, Ordering = list(map(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x, [Registry, Ordering]))
            charmapType = Registry + '-' + Ordering
            if charmapType in ['Adobe-Identity', 'Adobe-UCS'] and self.embedFont:
                self.charmap.update(self.embedFont.get_charmap())
            else:
                self.charmap = cidsystem.getCMap(
                    charmapType, cmap=self.charmap)

    def readMetrics(self, spec):
        dw_key = 'DW' if not self.vertical else 'DW2'
        ws_key = 'W' if not self.vertical else 'W2'

        if dw_key in spec:
            dw = spec[dw_key]
            _lsb = 0
            if isinstance(dw, (list, tuple)):
                _lsb, dw = dw
                _lsb = (1000 - int(_lsb)) * 0.001
            dw = int(dw) * 0.001
            self.metrics.update({cid: (dw, _lsb or self.metrics[cid][1])
                                 for cid in self.metrics})

        widths = {}
        r = []
        max_r = 5 if self.vertical else 3
        for x in list_value(spec.get(ws_key, [])):
            if isinstance(x, list):
                if not r:
                    continue
                x = [x[i:i+3]
                     for i in range(0, len(x), max_r)] if self.vertical else x
                for (i, y) in enumerate(x):
                    widths[r[-1]+i] = (y[0], y[1:]) if self.vertical else y
                r = []
            elif isinstance(x, (int, float)):
                r.append(x)
                if len(r) < max_r:
                    continue
                y = r[2:]
                for i in range(r[0], r[1]+1):
                    widths[i] = (y[0], y[1:]) if self.vertical else y[0]
                r = []

        for cid in widths:
            lsb = 0
            mtx = self.metrics[cid]
            if mtx:
                adw, lsb = mtx
            width = widths[cid]
            if not isinstance(width, int):
                adw = abs(width[0]) * 0.001
                lsb = 1 - width[1][1] * 0.001
            else:
                adw = width * 0.001
            self.metrics[cid] = (adw, lsb)

    def decode(self, bytes):
        return self.charset.decode(bytes)

    def to_unicode(self, cid):
        return self.charmap.get(cid) or '\ufffd'


class Type1Font(SimpleFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return T1FontFile(obj)


class Type1CFont(SimpleFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return CFF_FontFile(obj)


class TrueTypeFont(SimpleFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return TTFontFile(obj)


class Type3Font(TrueTypeFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)
        self.charProcs = dict_value(spec['CharProcs'])
        self.resources = dict_value(spec['Resources'])


class CIDType0Font(CIDFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return T1FontFile(obj)


class CIDType0CFont(CIDFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return CFF_FontFile(obj)


class CIDType2Font(CIDFont):
    def __init__(self, spec, embedFont=None, opentype=False):
        super().__init__(spec, embedFont=embedFont, opentype=opentype)

    def readEmbedFont(self, obj):
        return TTFontFile(obj)


def makeFont(spec):
    fontType, embedFont, opentype = getType(spec)
    if not fontType:
        return
    return fontType(spec, embedFont=embedFont, opentype=opentype)


types = {
    "Type1": Type1Font,
    "MMType1": Type1Font,
    "Type1C": Type1CFont,
    "Type3": Type3Font,
    "TrueType": TrueTypeFont
}
types0 = {
    "Type1": CIDType0Font,
    "Type1C": CIDType0CFont,
    "TrueType": CIDType2Font,
    "CIDFontType0": CIDType0Font,
    "CIDFontType0C": CIDType0CFont,
    "CIDFontType2": CIDType2Font
}


def getType(spec):
    if not spec:
        return spec, None, None, False

    isType0 = False
    embedFont = None
    opentype = False

    subtype = literal_name(spec['Subtype'])
    fontType = types.get(subtype)
    _spec = spec.copy()

    isType0 = subtype == "Type0"
    if isType0:
        dfonts = list_value(spec['DescendantFonts'])
        assert dfonts
        spec = dict_value(dfonts[0])
        subtype = literal_name(spec['Subtype'])
        fontType = types0.get(subtype)

    if 'FontDescriptor' in spec:
        descriptor = dict_value(spec['FontDescriptor'])
        for key in ['FontFile', 'FontFile2', 'FontFile3']:
            if key in descriptor:
                embedFont = stream_value(descriptor[key])
                if key == 'FontFile':
                    fontType = Type1Font
                elif key == 'FontFile2':
                    fontType = CIDType2Font if isType0 else TrueTypeFont
                elif key == 'FontFile3':
                    subtype = literal_name(embedFont.get('Subtype'))
                    fontType = (types0 if isType0 else types).get(subtype)
                    opentype = subtype == 'Opentype'
                break

    if embedFont:
        data = embedFont.get_data()
        subtype = identifyType(data)
        if subtype.endswith('-Opentype'):
            opentype = True
            subtype = subtype[:-9]
        fontType = types0.get(subtype) if isType0 else types.get(subtype)

    spec = dict(spec, **_spec)
    return spec, fontType, embedFont, opentype


def identifyType(data):
    def checkT1F(data):
        for head in [b'%!PS-AdobeFont', b'%!FontType1']:
            if data[:len(head)] == head:
                return 'Type1'

    def checkTTF(data):
        if data[:4] in [b'\x00\x01\x00\x00', b'true', b'ttcf']:
            return 'TrueType'

    def checkOTF(data):
        if data[:4] == b'OTTO':
            numTables, = struct.unpack('H', data[4:])
            entryFormat = '>4sLLL'
            entryFormatSize = struct.calcsize(entryFormat)
            for i in range(numTables):
                j = 12 + i * entryFormatSize
                tag, _, offset, __ = struct.unpack(
                    entryFormat, data[j:j+entryFormatSize])
                if tag == b'CFF ':
                    subtype = checkCFF(data, offset=offset)
                    if subtype:
                        return subtype+'-Opentype'

    def checkCFF(data, offset=0):
        data = data[offset:]
        if data[:2] != b'\x01\x00':
            if not offset:
                return checkCFF(data, offset=1)
            return
        hdrSize = data[2]
        offSize = data[3]
        if hdrSize < 0 or offSize < 1 or offSize > 4:
            return

        data = data[hdrSize:]
        count, = struct.unpack('>H', data[:2])
        offSize = data[2]
        if count < 0 or offSize < 1 or offSize > 4:
            return
        data = data[3:]
        padding = 0
        def pad(offSize): return b'\0' * (4 - offSize)
        offsets = []
        for i in range(count + 1):
            chunk = data[i*offSize:(i+1)*offSize]
            chunk = pad(offSize) + chunk
            offset, = struct.unpack(">L", chunk)
            offsets.append(int(offset))
            padding += offSize
        padding = padding - 1 + offsets[-1]
        data = data[padding:]

        count, = struct.unpack('>H', data[:2])
        offSize = data[2]
        if count < 0 or offSize < 1 or offSize > 4:
            return
        data = data[3:]
        offset0, = struct.unpack(">L", pad(offSize) + data[:offSize])
        offset1, = struct.unpack(">L", pad(offSize) + data[offSize:offSize*2])
        pos = (count + 1) * offSize + int(offset0) - 1
        endPos = (count + 1) * offSize + int(offset1) - 1

        data = data[pos:]
        for i in range(3):
            b = data[pos]
            if b == 0x1c:
                pos += 2
            elif b == 0x1d:
                pos += 4
            elif 0xf7 <= b <= 0xfe:
                pos += 1
            elif b < 0x20 or b > 0xf6:
                return 'Type1C'
            if pos >= endPos:
                return 'Type1C'
        if pos + 1 < endPos and data[:2] == b'\x0c\x1e':
            return "CIDFontType0C"
        else:
            return 'Type1C'

    identify_list = [checkT1F, checkTTF, checkOTF, checkCFF]

    for check_func in identify_list:
        subtype = check_func(data)
        if subtype:
            return subtype
