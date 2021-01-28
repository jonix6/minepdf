
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, PDFTextState, PDFGraphicState
from pdfminer.pdftypes import list_value, dict_value, stream_value, PDFStream
from pdfminer.psparser import LIT, PSLiteral
from pdfminer.pdftypes import PDFObjRef, resolve1
from pdfminer.utils import mult_matrix

from pdftext import TextAnalyzer, textSpanBox
import pdffonts
import colorspace


def literal(name): return LIT(
    name) if not isinstance(name, PSLiteral) else name


def render_type(ftype):
    def render_function(func):
        def render_arguments(self, *args, **kwargs):
            if ftype in self.filtered:
                return
            return func(self, *args, **kwargs)
        return render_arguments
    return render_function


def get_default(res_type):
    def binding(func):
        def get_arguments(self, objid, obj=None):
            res_list = getattr(self, res_type+'s', None)
            if res_list is None:
                return
            if objid is not None:
                objid = literal(objid)
                if objid in res_list:
                    return res_list[objid]
                elif obj is None:
                    return
            func(self, objid, obj=obj)
            if objid is not None:
                return res_list.get(objid)
        return get_arguments
    return binding


class Paint:
    def __init__(self, cs, value):
        self.cs = cs
        self.value = value

    def draw(self):
        return self.cs.getRGB(*self.value)


class TextState(PDFTextState):
    def __init__(self):
        super().__init__()
        self.fill = None
        self.extState = {}

    def copy(self):
        obj = self.__class__()
        obj.font = self.font
        obj.fontsize = self.fontsize
        obj.charspace = self.charspace
        obj.wordspace = self.wordspace
        obj.scaling = self.scaling
        obj.leading = self.leading
        obj.render = self.render
        obj.rise = self.rise
        obj.matrix = self.matrix
        obj.linematrix = self.linematrix
        obj.fill = self.fill
        obj.extState = self.extState
        return obj

    def __setattr__(self, key, value):
        if key in ['charspace', 'wordspace']:
            value *= getattr(self, 'scaling', 100) * 0.01
        return object.__setattr__(self, key, value)


class GraphicState(PDFGraphicState):
    def __init__(self):
        super().__init__()
        self.stroke = self.fill = None
        self.extState = {}

    def copy(self):
        obj = self.__class__()
        obj.linewidth = self.linewidth
        obj.linecap = self.linecap
        obj.linejoin = self.linejoin
        obj.miterlimit = self.miterlimit
        obj.dash = self.dash
        obj.intent = self.intent
        obj.flatness = self.flatness
        obj.stroke = self.stroke
        obj.fill = self.fill
        obj.extState = self.extState
        return obj


class Device(PDFDevice):
    def __init__(self, filtered=None, laparams=None, check_visible=True):
        super().__init__(None)
        self.filtered = filtered or []
        self.check_visible = check_visible
        self.analyzer = TextAnalyzer(**(laparams or {}))

        self.pageno = 1
        self.reset()
        self.viewBox = [0, 0, 0, 0]

    def reset(self):
        self.images = {}
        self.text_layer = []
        self.layers = {}
        self.layer_stack = []

    def begin_page(self, page, ctm):
        self.reset()
        self.layers[LIT('Page')] = (page.cropbox, ctm)
        self.layer_stack = [LIT('Page')]
        self.viewBox = page.cropbox
        self.ymax = page.mediabox[3] - page.mediabox[1]

    def is_visible(self, span, bbox):
        boxset = set(map(lambda p: (int(p[0]), int(p[1])), span.bbox))
        if len(boxset) < len(span.bbox):
            return False
        xmin, ymin, xmax, ymax = bbox
        return all(xmin < x < xmax and ymin < y < ymax for x, y in boxset)

    def get_current_layer(self):
        i = -1
        depth = 0
        while True:
            layerName = self.layer_stack[i]
            if layerName == 'end':
                depth += 1
            else:
                depth -= 1
            if depth < 0:
                break
            i -= 1
        return layerName, self.layers[layerName]

    def end_page(self, page):
        self.text_layer = filter(lambda x: not self.check_visible
                                 or self.is_visible(x, self.viewBox), self.text_layer)
        lines = self.analyzer.group_lines(self.text_layer)
        paras = self.analyzer.group_paras(lines)
        self.text_layer = paras
        self.pageno += 1

    def begin_figure(self, name, bbox, matrix):
        x, y, w, h = bbox
        self.layers[name] = ([x, y, x+w, y+h], matrix)
        self.layer_stack.append(name)

    def end_figure(self, name):
        self.layer_stack.append('end')

    @render_type('path')
    def paint_path(self, graphicstate, stroke, fill, evenodd, path):
        # path handling suspended
        return path

    @render_type('image')
    def render_image(self, name, stream, anchored=False, textstate=None):
        bbox, matrix = self.get_current_layer()[1]
        self.images.setdefault(stream.objid, (name, stream, bbox, matrix))

    @render_type('text')
    def render_string(self, textstate, seq, *args):
        layerName = self.get_current_layer()[0]
        x, y = textstate.linematrix
        a, b, c, d, e, f = mult_matrix(textstate.matrix, self.ctm)
        matrix = a, b, c, d, e, self.ymax - f
        box = textSpanBox((x, y), seq, textstate, layerName=layerName, matrix=matrix)

        # check if text is visible
        if not textstate.extState.get('OP', False) or not textstate.extState.get('OPM', 0):
            self.text_layer.append(box)
        elif textstate.extState.get('OPM', 1) and any(textstate.fill.value):
            self.text_layer.append(box)

        textstate.linematrix = box.originbox[2]


class ResourceManager(PDFResourceManager):
    def __init__(self):
        self.fonts = {}
        self.colorspaces = colorspace.defaults.copy()
        self.xobjects = {}

        self.cache = {}
        self.stream_objects = []

    def clear(self):
        for res in self.fonts:
            stream_to_close = getattr(res, 'embedFont', None)
            stream_to_close and stream_to_close.close()
        self.fonts.clear()
        self.colorspaces.clear()
        self.xobjects.clear()

    def render_resource(self, res_type, res_obj):
        get_function = getattr(self, 'get_' + res_type.lower(), None)
        return get_function and get_function(None, obj=res_obj)

    @get_default('font')
    def get_font(self, objid, obj=None):
        for (fontid, spec) in dict_value(obj).items():
            spec = dict_value(spec)
            spec, fontType, embedFont, opentype = pdffonts.getType(spec)
            if fontType:
                font = fontType(spec, embedFont=embedFont and self.xobjects.get(
                    embedFont.objid, embedFont), opentype=opentype)
                if embedFont:
                    objid = literal(embedFont.objid)
                    if not objid in self.xobjects:
                        self.xobjects[objid] = font.embedFont
                self.fonts[literal(fontid)] = font

    @get_default('colorspace')
    def get_colorspace(self, objid, obj=None):
        for (csid, spec) in dict_value(obj).items():
            cs = colorspace.parse(spec)
            if cs:
                self.colorspaces[literal(csid)] = cs

    def get_procset(self, objid, obj=None):
        # procset handling suspended
        pass

    @get_default('xobject')
    def get_xobject(self, objid, obj=None):
        for (xobjid, xobjstrm) in dict_value(obj).items():
            self.xobjects[literal(xobjid)] = xobjstrm


class Interpreter(PDFPageInterpreter):
    def __init__(self, device):
        self.rsrcmgr = ResourceManager()
        self.device = device

    # custom logging here
    def log(self, message):
        pass

    def dup(self):
        return self.__class__(self.device)

    def close(self):
        self.rsrcmgr.clear()

    def init_resources(self, resources):
        self.resources = resources
        if resources:
            for (k, v) in dict_value(resources).items():
                self.debug and self.log('Resource: %r: %r' % (k, v))
                self.rsrcmgr.render_resource(k, v)

    def init_state(self, ctm):
        self.gstack = []
        self.ctm = ctm
        self.device.set_ctm(self.ctm)
        self.textstate = TextState()
        self.graphicstate = GraphicState()
        self.curpath = []
        self.argstack = []
        self.scs = self.ncs = colorspace.CMYKColorSpace()

    def do_CS(self, name):
        self.scs = self.rsrcmgr.get_colorspace(literal(name))

    def do_cs(self, name):
        self.ncs = self.rsrcmgr.get_colorspace(literal(name))

    def do_SCN(self):
        n = len(self.scs.mode)
        pattern = self.argstack[-n:]
        self.graphicstate.stroke = Paint(self.scs, pattern)
        self.argstack = self.argstack[:-n]

    def do_scn(self):
        n = len(self.ncs.mode)
        pattern = self.argstack[-n:]
        self.graphicstate.fill = self.textstate.fill = Paint(self.ncs, pattern)
        self.argstack = self.argstack[:-n]

    def do_G(self, gray):
        cs = colorspace.GrayColorSpace()
        self.graphicstate.stroke = Paint(cs, gray)

    def do_g(self, gray):
        cs = colorspace.GrayColorSpace()
        self.graphicstate.fill = self.textstate.fill = Paint(cs, gray)

    def do_RG(self, r, g, b):
        cs = colorspace.RGBColorSpace()
        self.graphicstate.stroke = Paint(cs, (r, g, b))

    def do_rg(self, r, g, b):
        cs = colorspace.RGBColorSpace()
        self.graphicstate.fill = self.textstate.fill = Paint(cs, (r, g, b))

    def do_K(self, c, m, y, k):
        cs = colorspace.CMYKColorSpace()
        self.graphicstate.stroke = Paint(cs, (c, m, y, k))

    def do_k(self, c, m, y, k):
        cs = colorspace.CMYKColorSpace()
        self.graphicstate.fill = self.textstate.fill = Paint(cs, (c, m, y, k))

    def do_Tf(self, fontid, fontsize):
        self.textstate.font = self.rsrcmgr.get_font(literal(fontid))
        self.textstate.fontsize = fontsize

    def do_Do(self, xobjid):
        xobj = self.rsrcmgr.get_xobject(literal(xobjid))
        if not xobj:
            return
        self.debug and self.log('Processing xobj: %r' % xobj)
        xobj = stream_value(xobj)
        subtype = xobj.get('Subtype')
        if subtype is LIT('Form') and 'BBox' in xobj:
            interpreter = self.dup()
            bbox = list_value(xobj['BBox'])
            matrix = list_value(xobj.get('Matrix', (1, 0, 0, 1, 0, 0)))
            # According to PDF reference 1.7 section 4.9.1, XObjects in
            # earlier PDFs (prior to v1.2) use the page's Resources entry
            # instead of having their own Resources entry.
            resources = dict_value(xobj.get('Resources')
                                   ) or self.resources.copy()
            self.device.begin_figure(xobjid, bbox, matrix)
            interpreter.render_contents(
                resources, [xobj], ctm=mult_matrix(matrix, self.ctm))
            self.device.end_figure(xobjid)
        elif subtype is LIT('Image') and 'Width' in xobj and 'Height' in xobj:
            self.device.render_image(xobjid, xobj, anchored=True)
        else:
            # unsupported xobject type.
            pass

    def do_EI(self, obj):
        if 'W' in obj and 'H' in obj:
            self.device.render_image(
                str(id(obj)), obj, anchored=False, state=self.textstate)

    def do_gs(self, name):
        if isinstance(name, PSLiteral):
            name = name.name
        gstate = self.resources['ExtGState'].get(name)
        if gstate and not self.textstate.extState:
            gstate = resolve1(gstate)
            self.textstate.extState = gstate

    def do_q(self):
        self.gstack.append(self.get_current_state())

    def do_Q(self):
        self.gstack and self.set_current_state(self.gstack.pop())

    # def do_Td(self, tx, ty):
    # 	x, y = self.textstate.linematrix
    # 	# print((x,y), (tx,ty))
    # 	(a, b, c, d, e, f) = self.textstate.matrix
    # 	print((x,y), (tx,ty), (tx*a+ty*c+e, tx*b+ty*d+f))
    # 	self.textstate.matrix = (a, b, c, d, tx*a+ty*c+e, tx*b+ty*d+f)
    # 	self.textstate.linematrix = (0, 0)

