
import struct


def tohalf(code):
    ffe0 = [0xa2, 0xa3, 0xac, 0xaf, 0xa6, 0xa5, 0x20a9]
    if code <= 0xff00:
        return code
    if 0xff01 <= code <= 0xff5e:
        code -= 0xfee0
    elif 0xff5f <= code <= 0xff60:
        code = 0x2985 + code - 0xff5f
    elif 0xffe0 <= code <= 0xffe6:
        code = ffe0[code - 0xffe0]
    return code


class DECODER:
    @property
    def glyphlist(self):
        data = open('./cidToUnicode/private.dat', 'rb')
        e000 = []
        while 1:
            chunk = data.read(2)
            if not chunk:
                break
            code, = struct.unpack('>H', chunk)
            e000.append(code)
        data.close()
        return e000

    def valid_unicode(self, code):
        if 0xe000 <= code <= 0xe814:
            code = self.glyphlist[code - 0xe000]
        return chr(code)

    def decode(self, bytes):
        return bytes.decode('gb18030')


class DECODER_PK748_E(DECODER):
    def decode(self, bytes):
        code, = struct.unpack('>H', bytes)
        if code < 0x1000:
            return chr(code)
        hb, lb = struct.unpack('>2B', struct.pack('>H', code))
        if not 0xa0 <= hb <= 0xad:
            return ord(struct.pack('>2B', hb, lb).decode('gb18030'))

        if hb == 0xa0 and 0x80 <= lb <= 0xfe:
            if lb < 0xfe:
                hb += 3
            else:
                hb, lb = 0xa1, 0xab
        elif hb == 0xa2 and 0x41 <= lb <= 0x7f:
            lb += 0x1f
        elif hb == 0xa6 and 0x40 <= lb <= 0x5f:
            lb += 0x1f

        code = ord(struct.pack('>2B', hb, lb).decode('gb18030'))
        return self.valid_unicode(tohalf(code))


def encoding(enc_type):
    assert enc_type.startswith('Founder-')
    enc_type = enc_type[8:].lower()
    if enc_type == 'pkue1':
        return DECODER_PK748_E()
    return DECODER()
