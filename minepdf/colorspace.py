
from pdfminer.psparser import LIT, PSLiteral, PSStackParser, PSKeyword, PSEOF, keyword_name
from pdfminer.pdftypes import PDFObjRef, resolve1, dict_value, stream_value, list_value, PDFStream

from PIL import ImageCms
from io import BytesIO
import numpy as np
from itertools import product


class colorSpaces:
    @property
    def defaults(self):
        default_values = [
            (GrayColorSpace, LIT('DeviceGray'), LIT('G')),
            (RGBColorSpace, LIT('DeviceRGB'), LIT('RGB')),
            (CMYKColorSpace, LIT('DeviceCMYK'), LIT('CMYK')),
            (CalGrayColorSpace, LIT('CalGray')),
            (CalRGBColorSpace, LIT('CalRGB')),
            (LabColorSpace, LIT('Lab')),
            (ICCBasedColorSpace, LIT('ICCBased')),
            (IndexedColorSpace, LIT('Indexed')),
            (SeparationColorSpace, LIT('Separation')),
            # (DeviceNColorSpace, LIT('DeviceN')),
            (PatternColorSpace, LIT('Pattern')),
            (NColorSpace, LIT('DeviceN')),
        ]
        refs = {}
        for tpl in default_values:
            for i, x in enumerate(tpl):
                if i > 0:
                    refs[x] = tpl[0]
        return refs

    def parse(self, obj, args=[]):
        if isinstance(obj, PDFObjRef):
            obj = resolve1(obj)

        if isinstance(obj, PSLiteral):
            cs = self.defaults.get(obj)
            if not cs:
                return None
                # raise TypeError('unknown color space: %s' % obj.name)
            return cs(*args)

        if isinstance(obj, list):
            return self.parse(obj[0], args=obj[1:])


class ColorSpace:
    overprintMask = 0x0f
    pipe = lambda *val: val
    getGray = pipe
    getRGB = pipe
    getCMYK = pipe
    mapGray = pipe
    mapRGB = pipe
    mapCMYK = pipe


class GrayColorSpace(ColorSpace):
    mode = 'L'
    ncomps = 1

    def getRGB(self, gray):
        # [gray] · [1, 1, 1]
        r = g = b = gray
        return r, g, b

    def getCMYK(self, gray):
        # [gray] · [0, 0, 0, 1]
        c = m = y = 0
        k = gray
        return c, m, y, k


class CalGrayColorSpace(GrayColorSpace):
    whiteX = whiteY = whiteZ = 1
    blackX = blackY = blackZ = 0
    gamma = 1

    def __init__(self, obj):
        obj = resolve1(obj)
        params = dict_value(obj)
        self.whiteX, self.whiteY, self.whiteZ = params['WhitePoint']
        self.blackX, self.blackY, self.blackZ = params['BlackPoint']
        self.gamma = params['Gamma']


class RGBColorSpace(ColorSpace):
    mode = 'RGB'
    ncomps = 3

    def getGray(self, r, g, b):
        return 0.299 * r + 0.587 * g + 0.114 * b

    def getCMYK(self, r, g, b):
        c = 1 - r
        m = 1 - g
        y = 1 - b
        k = min(c, m, y)
        return c - k, m - k, y - k, k

    def mapGray(self, arr):
        return self.getGray(arr[..., 0], arr[..., 1], arr[..., 2])

    def mapCMYK(self, arr):
        k = arr.max(-1)
        out = np.empty_like(arr)
        out[..., 0] = k - arr[..., 0]
        out[..., 1] = k - arr[..., 1]
        out[..., 2] = k - arr[..., 2]
        k = k[..., np.newaxis]
        return np.concatenate((out, 255 - k), axis=-1)


class CalRGBColorSpace(RGBColorSpace):
    matrix = [
        1,	0,	0,
        0,	1,	0,
        0,	0,	1
    ]

    def __init__(self, obj):
        obj = resolve1(obj)
        params = dict_value(obj)
        self.whiteX, self.whiteY, self.whiteZ = params.get(
            'WhitePoint', (1, 1, 1))
        self.blackX, self.blackY, self.blackZ = params.get(
            'BlackPoint', (0, 0, 0))
        self.gammaR, self.gammaG, self.gammaB = params.get('Gamma', (1, 1, 1))
        self.matrix = params.get('Matrix', self.matrix)


class CMYKColorSpace(ColorSpace):
    mode = 'CMYK'
    ncomps = 4
    factors = [
        [1,		 1,	     1],
        [0.1373, 0.1216, 0.1255],
        [1,		 0.9490, 0],
        [0.1098, 0.1020, 0],
        [0.9255, 0,	     0.5490],
        [0.1412, 0,	     0],
        [0.9294, 0.1098, 0.1412],
        [0.1333, 0,	     0],
        [0,		 0.6784, 0.9373],
        [0,		 0.0588, 0.1412],
        [0,		 0.6510, 0.3137],
        [0,		 0.0745, 0],
        [0.1804, 0.1922, 0.5725],
        [0,		 0,		 0.0078],
        [0.2118, 0.2119, 0.2235],
        [0,		 0,	     0]
    ]

    def getGray(self, c, m, y, k):
        return 1 - k - 0.3 * c - 0.59 * m - 0.11 * y

    def getRGB(self, c, m, y, k, r=0, g=0, b=0):
        c1, m1, y1, k1 = 1-c, 1-m, 1-y, 1-k
        for i, (b0, b1, b2, b3) in enumerate(product([c1, c], [m1, m], [y1, y], [k1, k])):
            x = b0 * b1 * b2 * b3
            r += self.factors[i][0] * x
            g += self.factors[i][1] * x
            b += self.factors[i][2] * x
        return r, g, b

    def mapGray(self, arr):
        return 255 - arr[..., 3] - 0.3 * arr[..., 0] - 0.59 * arr[..., 1] - 0.11 * arr[..., 2]

    def mapRGB(self, arr):
        arr = arr.astype('float') / 255
        out = np.empty_like(arr[..., :-1], dtype='float')
        self.getRGB(*(arr[..., i] for i in range(4)),
                    *(out[..., i] for i in range(3)))
        return (out * 255).astype('uint8')


xyzrgb = [
    [3.240449,	-1.537136,	-0.498531],
    [-0.969265,	1.876011,	0.041556],
    [0.055643,	-0.204026,	1.057229]
]


class LabColorSpace(ColorSpace):
    mode = 'LAB'
    ncomps = 3

    def __init__(self, obj):
        obj = resolve1(obj)
        params = dict_value(obj)
        self.whiteX, self.whiteY, self.whiteZ = params.get(
            'WhitePoint', (1, 1, 1))
        self.blackX, self.blackY, self.blackZ = params.get(
            'BlackPoint', (0, 0, 0))
        self.aMin, self.bMin, self.aMax, self.bMax = params.get(
            'Range', (-100, -100, 100, 100))
        self.kr = 1 / (
            xyzrgb[0][0] * self.whiteX +
            xyzrgb[0][1] * self.whiteY +
            xyzrgb[0][2] * self.whiteZ
        )
        self.kg = 1 / (
            xyzrgb[1][0] * self.whiteX +
            xyzrgb[1][1] * self.whiteY +
            xyzrgb[1][2] * self.whiteZ
        )
        self.kb = 1 / (
            xyzrgb[2][0] * self.whiteX +
            xyzrgb[2][1] * self.whiteY +
            xyzrgb[2][2] * self.whiteZ
        )

    def getGray(self, l, a, b):
        r, g, b = self.getRGB(l, a, b)
        return 0.299 * r + 0.587 * g + 0.114 * b + 0.5

    def getRGB(self, l, a, b):
        def lab2xyz(t): return t ** 3 if (t >= 6 /
                                          29) else (108 / 841 * (t - 4 / 29))

        # convert L*a*b* to CIE 1931 XYZ color space
        t1 = (l + 16) / 116
        t2 = t1 + a / 500
        X = lab2xyz(t2)
        X *= self.whiteX
        Y = lab2xyz(t1)
        Y *= self.whiteY
        t2 = t1 - b / 200
        Z = lab2xyz(t2)
        Z *= self.whiteZ

        # convert XYZ to RGB, including gamut mapping and gamma correction
        r = xyzrgb[0][0] * X + xyzrgb[0][1] * Y + xyzrgb[0][2] * Z
        g = xyzrgb[1][0] * X + xyzrgb[1][1] * Y + xyzrgb[1][2] * Z
        b = xyzrgb[2][0] * X + xyzrgb[2][1] * Y + xyzrgb[2][2] * Z

        return r ** 0.5, g ** 0.5, b ** 0.5

    def getCMYK(self, l, a, b):
        r, g, b = self.getRGB(l, a, b)
        c = 1 - r
        m = 1 - g
        y = 1 - b
        k = min(c, m, y)
        return c - k, m - k, y - k, k


class ICCBasedColorSpace(ColorSpace):
    @property
    def defaults(self):
        return {
            'L': GrayColorSpace,
            'RGB': RGBColorSpace,
            'CMYK': CMYKColorSpace,
            'LAB': LabColorSpace
        }

    mode = 'RGB'

    def __init__(self, obj):
        obj = resolve1(obj)
        fp = BytesIO(obj.get_data())
        self.profile = ImageCms.ImageCmsProfile(fp)
        fp.close()
        self.mode = self.profile.profile.color_space
        if self.mode == 'LAB':
            alt = resolve1(obj['Alternate'])
            if isinstance(alt, list):
                alt = alt[1]
            self.base = self.defaults[self.mode](alt)
        else:
            self.base = self.defaults[self.mode]()
        self.ncomps = len(self.mode)

    def getGray(self, *val):
        return self.base.getGray(*val)

    def getRGB(self, *val):
        return self.base.getRGB(*val)

    def getCMYK(self, *val):
        return self.base.getCMYK(*val)


class IndexedColorSpace(ColorSpace):
    mode = 'P'
    basemode = 'RGB'
    palette = list(map(lambda i: (i, i, i), range(256)))
    ncomps = 1

    def __init__(self, base, hival, obj):
        cs = colorSpaces()
        self.base = cs.parse(resolve1(base))
        self.hival = int(resolve1(hival))

        obj = resolve1(obj)
        data = b''
        if isinstance(obj, bytes):
            data = obj
        elif isinstance(obj, PDFStream):
            data = obj.get_data()
        if data:
            n = self.base.ncomps
            self.palette = [[data[i * n + j] for j in range(n)] for i in range(len(data) // n)]

    def lookup(self, index):
        i = max(0, min(index, len(self.palette) - 1))
        return self.palette[i]

    def getGray(self, index):
        return self.base.getGray(*self.lookup(index))

    def getRGB(self, index):
        return self.base.getRGB(*self.lookup(index))

    def getCMYK(self, index):
        return self.base.getCMYK(*self.lookup(index))

    def mapPixels(self, arr):
        palette = np.array(self.palette, dtype='uint8')
        return palette[arr]

    def mapGray(self, arr):
        return self.base.mapGray(arr)

    def mapRGB(self, arr):
        return self.base.mapRGB(arr)

    def mapCMYK(self, arr):
        return self.base.mapCMYK(arr)


class functionParser:
    def _min(self, x, num):
        if isinstance(x, (int, float)):
            return min(x, num)
        x[x >= num] = num
        return x

    def _max(self, x, num):
        if isinstance(x, (int, float)):
            return max(x, num)
        x[x < num] = num
        return x


class SampledFunctionParser(functionParser):
    def __init__(self, spec, domain):
        self.domain = domain
        self.frange = list_value(spec['Range'])
        self.nins = len(self.domain) >> 1
        self.nouts = len(self.frange) >> 1

        self.sizes = list_value(spec['Size'])[:self.nins]
        self.bits = int(spec['BitsPerSample'])

        if 'Encode' in spec:
            self.encode = list_value(spec['Encode'])
        else:
            self.encode = [0] * (self.nins << 1)
            self.encode[1::2] = [size-1 for size in self.sizes]

        self.decode = list_value(
            spec['Decode']) if 'Decode' in spec else self.frange[:]

        # domain = [0 1]
        # range = [0 1 0 1 0 1 0 1]
        # bits = 8
        # sizes = [1024]
        # encode = [0 1023]
        # decode = [0 1 0 1 0 1 0 1]

    def interpolate(self, x, xmin, xmax, ymin, ymax):
        return (ymax - ymin) / (xmax-xmin) * (x-xmin) + ymin

    def parse(self, *args):
        e = []
        for i in range(self.nins):
            x = self._min(
                self._max(args[i], self.domain[i*2]), self.domain[i*2+1])
            x = self.interpolate(
                x, self.domain[i*2], self.domain[i*2+1], self.encode[i*2], self.encode[i*2+1])
            e.append(self._min(self._max(x, 0), self.sizes[i]-1))
        return e


def SampledFunction(spec, domain):
    parser = SampledFunctionParser(spec, domain)
    return parser.parse


class ExponentialFunctionParser(functionParser):
    def __init__(self, spec, domain):
        self.c0, self.c1 = [0], [1]
        if spec.get('C0'):
            self.c0 = [float(x) for x in list_value(spec['C0'])]
        if spec.get('C1'):
            self.c1 = [float(x) for x in list_value(spec['C1'])]
        self.n = spec['N']
        self.frange = None
        if spec.get('Range'):
            self.frange = list_value(spec.get('Range'))
        self.domain = domain

    def parse(self, ipt):
        ipt /= 255
        ipt = self._min(self._max(ipt, self.domain[0]), self.domain[1])
        opt = []
        for i in range(len(self.c0)):
            x = self.c0[i] + pow(ipt, self.n) * (self.c1[i] - self.c0[i])
            if self.frange:
                x = self._min(self._max(x, self.frange[0]), self.frange[1])
            opt.append(x * 255)
        return opt


def ExponentialFunction(spec, domain):
    parser = ExponentialFunctionParser(spec, domain)
    return parser.parse


def StitchingFunction(spec, domain):
    pass


class PSFunctionParser(PSStackParser):
    def __init__(self, fp):
        super().__init__(fp)
        self.run()

    def run(self):
        try:
            self.nextobject()
        except PSEOF:
            pass
        _, self.argstack = self.curstack.pop()
        self.reset()

    def parse(self, *args):
        argstack = list(args) + self.argstack
        self.curstack = []
        while argstack:
            obj = argstack.pop(0)
            if isinstance(obj, PSKeyword):
                name = keyword_name(obj)
                if not isinstance(name, str):
                    name = name.decode()
                result = getattr(self, 'do_'+name)()
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        self.curstack += list(result)
                    else:
                        self.curstack.append(result)
            else:
                self.curstack.append(obj)
        return self.curstack

    def do_keyword(self, pos, token):
        self.push((pos, token))

    def do_roll(self):
        n, j = self.pop(2)
        vals = self.pop(n)
        j %= n
        if not j:
            return vals
        return (vals*2)[n-j:n*2-j]

    def do_dup(self):
        x = self.pop(1)
        return x + x

    def do_exch(self):
        a, b = self.pop(2)
        return b, a

    def do_sub(self):
        a, b = self.pop(2)
        if isinstance(b, (int, float)):
            return b - a
        b[b < a] = 0
        b[b >= a] -= a
        return b

    def do_pop(self):
        self.pop(1)

    def do_index(self):
        i = self.pop(1)[0]
        return self.curstack[-i-1]

    def do_cvr(self):
        num = self.pop(1)[0]
        return float(num)

    def do_mul(self):
        a, b = self.pop(2)
        return a * b


def PostScriptFunction(spec, domain):
    parser = PSFunctionParser(BytesIO(spec.get_data()))
    return parser.parse


def func_parse(spec):
    func_type = int(spec.get('FunctionType'))
    domain = list_value(spec.get('Domain'))

    func_refs = {
        0: SampledFunction,
        2: ExponentialFunction,
        3: StitchingFunction,
        4: PostScriptFunction
    }

    func_builder = func_refs[func_type]
    return func_builder(spec, domain)


class SeparationColorSpace(ColorSpace):
    mode = 'P'

    def __init__(self, alt, base, func, *args):
        cs = colorSpaces()
        self.base = cs.parse(resolve1(base))
        spec = resolve1(func)
        self.ncomps = len(spec['Domain']) >> 1
        self.func = func_parse(spec)

    def transform(self, *val):
        transformed = self.func(*val)
        new_val = []
        for i in range(self.base.ncomps):
            new_val.append(transformed[i])
        return new_val

    def mapPixels(self, arr):
        if not self.func:
            return arr
        if len(arr.shape) == 2:
            arr = arr[..., np.newaxis]
        w, h, d = arr.shape
        arr = arr.astype('float')
        transformed = self.transform(*[arr[..., i] for i in range(d)])
        result = None
        for layer in transformed:
            if isinstance(layer, (int, float)):
                layer = np.ones((w, h), dtype='float') * layer
            layer = layer.astype('uint8')
            if result is None:
                result = layer
            else:
                result = np.dstack([result, layer])
        return result

    def getGray(self, *val):
        val = self.transform(*val)
        return self.base.getGray(*val)

    def getRGB(self, *val):
        val = self.transform(*val)
        return self.base.getRGB(*val)

    def getCMYK(self, *val):
        val = self.transform(*val)
        return self.base.getCMYK(*val)

    def mapGray(self, arr):
        return self.base.mapGray(arr)

    def mapRGB(self, arr):
        return self.base.mapRGB(arr)

    def mapCMYK(self, arr):
        return self.base.mapCMYK(arr)


class NColorSpace(SeparationColorSpace):
    mode = 'P'

    def __init__(self, names, alt, func, *attrs):
        self.names = list_value(names)
        self.base = colorSpaces().parse(resolve1(alt))
        spec = resolve1(func)
        self.ncomps = len(spec['Domain']) >> 1
        self.func = func_parse(spec)


class PatternColorSpace(ColorSpace):
    under = None
    mode = 'P'
    ncomps = 1

    def __init__(self, *args):
        if args:
            cs = colorSpaces()
            self.under = cs.parse(resolve1(args[0]))


defaults = colorSpaces().defaults
parse = colorSpaces().parse


