
import re
from collections import OrderedDict
import struct
import os
import decoder748

REG_EXP = re.compile(r'^\s*<([0-9a-f]+)>\s+<([0-9a-f]+)>\s+(\d+)$', re.M)


class CMap:
    MAP_STRING = ''

    def __init__(self):
        self.codePoints = set()
        self.cid2unicode = {}
        self._feed()

    def _feed(self):
        for (s, e, code) in re.findall(REG_EXP, self.MAP_STRING):
            s = int(s, 16)
            e = int(e, 16)
            self.codePoints.add(s)
            self.cid2unicode[s] = int(code)

    def to_unicode(self, cid):
        for point in self.codePoints:
            if cid <= point:
                break
        d = cid - point
        code = self.cid2unicode[point]
        return chr(code + d)


def to_unicode(klass, cid):
    if cid in klass.diff:
        return klass.diff[cid]
    point = 0
    for next_point in sorted(klass.cid2unicode.keys()):
        if cid < next_point:
            break
        point = next_point
    e = cid - point
    code = klass.cid2unicode[point] + e
    if code < 0x100:
        c = chr(code)
    elif code < 0x10000:
        c = struct.pack('>H', code).decode('gb18030')
    else:
        c = struct.pack('>L', code).decode('gb18030')
    return c


def to_unicode_wrapper(klass):
    def func(cid):
        return to_unicode(klass, cid)
    return func


class UnicodeMap:
    @property
    def DESC(self):
        return './cidtounicode'

    def __init__(self, cmap={}):
        self.cid2unicode = {}
        self.diff = cmap

    def get(self, cid):
        if cid in self.diff:
            return self.diff[cid]
        return chr(cid)


class ADOBE_GB1(UnicodeMap):
    FILE_NAME = 'Adobe-GB1.cidToUnicode'


def getCMap(cmapType, cmap={}):
    if cmapType.startswith('Founder-') and cmapType.endswith('748'):
        decoder = decoder748.encoding(cmapType)
        for cid in cmap:
            cmap[cid] = decoder.decode(cmap[cid].encode('gb18030'))
    elif cmapType == 'Adobe-GB1':
        cmap = ADOBE_GB1(cmap=cmap)
    return cmap
