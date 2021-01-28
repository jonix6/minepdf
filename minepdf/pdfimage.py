
from pdfminer.psparser import LIT
from pdfminer.ascii85 import ascii85decode, asciihexdecode
from pdfminer.lzw import lzwdecode
from pdfminer.runlength import rldecode

from PyPDF2.filters import FlateDecode
import colorspace

import struct
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageCms

MAXSIZE = 1080 * 1080

LITERAL_CRYPT = LIT('Crypt')
LITERALS_FLATE_DECODE = (LIT('FlateDecode'), LIT('Fl'))
LITERALS_LZW_DECODE = (LIT('LZWDecode'), LIT('LZW'))
LITERALS_ASCII85_DECODE = (LIT('ASCII85Decode'), LIT('A85'))
LITERALS_ASCIIHEX_DECODE = (LIT('ASCIIHexDecode'), LIT('AHx'))
LITERALS_RUNLENGTH_DECODE = (LIT('RunLengthDecode'), LIT('RL'))
LITERALS_CCITTFAX_DECODE = (LIT('CCITTFaxDecode'), LIT('CCF'))
LITERALS_DCT_DECODE = (LIT('DCTDecode'), LIT('DCT'))
LITERALS_JPX_DECODE = (LIT('JPXDecode'), LIT('JPX'))


def percent(x): return x / 255


def integer(x): return int(x * 255)


def image_decode(stream):
    data = stream.get_rawdata()
    filters = stream.get_filters()
    out_filters = []
    for (ft, params) in filters:
        while hasattr(ft, 'resolve'):
            ft = ft.resolve()
        if not ft:
            continue
        if ft in LITERALS_FLATE_DECODE:
            # flate decoder in pdfminer will raise exceptions, use pypdf2 instead
            data = FlateDecode.decode(data, params)
        elif ft in LITERALS_ASCIIHEX_DECODE:
            data = asciihexdecode(data)
        elif ft in LITERALS_LZW_DECODE:
            data = lzwdecode(data)
        elif ft in LITERALS_ASCII85_DECODE:
            data = ascii85decode(data)
        elif ft in LITERALS_RUNLENGTH_DECODE:
            data = rldecode(data)
        else:
            if ft == LITERAL_CRYPT:
                # not implemented yet
                raise Exception
            out_filters.append((ft, params))
    return data, out_filters


def tiff_header_for_CCITT(size, length, group=4):
    width, height = size
    tiff_header_struct = '<2shlh' + 'hhll' * 8 + 'h'
    return struct.pack(
        tiff_header_struct,
        b'II',  # Byte order indication: Little indian
        42,  # Version number (always 42)
        8,  # Offset to first IFD
        8,  # Number of tags in IFD
        256, 4, 1, width,  # ImageWidth, LONG, 1, width
        257, 4, 1, height,  # ImageLength, LONG, 1, lenght
        258, 3, 1, 1,  # BitsPerSample, SHORT, 1, 1
        259, 3, 1, group,  # Compression, SHORT, 1, 4 = CCITT Group 4 fax encoding
        262, 3, 1, 0,  # Threshholding, SHORT, 1, 0 = WhiteIsZero
        # StripOffsets, LONG, 1, len of header
        273, 4, 1, struct.calcsize(tiff_header_struct),
        278, 4, 1, height,  # RowsPerStrip, LONG, 1, lenght
        279, 4, 1, length,  # StripByteCounts, LONG, 1, size of image
        0  # last IFD
    )


def scale_image(im):
    if im.size[0] * im.size[1] > MAXSIZE:
        scale = (MAXSIZE / im.size[0] / im.size[1]) ** 0.5
        im = im.resize(
            (int(im.size[0]*scale), int(im.size[1]*scale)), Image.ANTIALIAS)
    return im


def get_image(stream):
    # if stream.get('ImageMask', False):
    #     return
    cs = colorspace.parse(stream.get_any(('CS', 'ColorSpace')))
    data, out_filters = image_decode(stream)
    assert len(out_filters) < 2

    size = (stream.get_any(('W', 'Width')), stream.get_any(('H', 'Height')))
    im = None
    if len(out_filters) == 0:
        im = Image.frombytes(cs.mode, size, data)
    else:
        ft, params = out_filters[0]
        if ft in LITERALS_DCT_DECODE:  # JPEG
            im = Image.open(BytesIO(data))
            if 'APP14' in im.app and im.app['APP14'][-1] == 2:
                # YCCK mode, color should be inverted
                im = ImageChops.invert(im)
        elif ft in LITERALS_JPX_DECODE:  # JPX
            im = Image.open(BytesIO(data))
        elif ft in LITERALS_CCITTFAX_DECODE:  # TIFF
            group = 4 if params['K'] == -1 else 3
            header = tiff_header_for_CCITT(size, len(data), group)
            im = Image.open(BytesIO(header+data))

    if not im:
        return
    if not cs:
        return scale_image(im)
    if hasattr(cs, 'profile'):
        return scale_image(ImageCms.profileToProfile(
            im, cs.profile, ImageCms.get_display_profile(), outputMode='RGB'))

    isArray = False
    while cs.mode == 'P':
        if hasattr(cs, 'mapPixels'):
            if not isArray:
                im = np.array(im)
            im = cs.mapPixels(im)
            isArray = True
        else:
            im = im.convert(cs.basemode)
        cs = cs.base

    if not isArray:
        if im.mode == 'CMYK' or cs.mode == 'CMYK':
            im = np.array(im).reshape((im.size[1], im.size[0], 4))
            im = colorspace.CMYKColorSpace().mapRGB(im)
            im = Image.fromarray(im, mode='RGB')
        return scale_image(im)

    mode = cs.mode
    if not mode.startswith(('L', 'RGB')):
        if not isArray:
            im = np.array(im).reshape((im.size[1], im.size[0], len(mode)))
        if hasattr(cs, 'profile'):
            im = ImageCms.profileToProfile(Image.fromarray(im, mode=mode),
                                           cs.profile, ImageCms.get_display_profile(), outputMode='RGB')
        else:
            im = cs.mapRGB(im)
            im = Image.fromarray(im, mode='RGB')
    elif isArray:
        im = Image.fromarray(im, mode=mode)

    return scale_image(im)
