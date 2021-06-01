from predify.modules import PCoder
from predify.networks import PNetSameHP, PNetSeparateHP
from torch.nn import Sequential, ConvTranspose2d, Upsample

class PEff_b0SeparateHP_V1(PNetSeparateHP):
    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier=(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3), fb_multiplier=(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3), er_multiplier=(0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01)):
        super().__init__(backbone, 8, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        # PCoder number 1
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear'),ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.act1.register_forward_hook(fw_hook1)

        # PCoder number 2
        pmodule = Sequential(ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)
        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.blocks[0].register_forward_hook(fw_hook2)

        # PCoder number 3
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear'),ConvTranspose2d(24, 16, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)
        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.blocks[1].register_forward_hook(fw_hook3)

        # PCoder number 4
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear'),ConvTranspose2d(40, 24, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoder(pmodule, True, self.random_init)
        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.blocks[2].register_forward_hook(fw_hook4)

        # PCoder number 5
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear'),ConvTranspose2d(80, 40, kernel_size=3, stride=1, padding=1))
        self.pcoder5 = PCoder(pmodule, True, self.random_init)
        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=self.pcoder6.prd, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm5, fbm=self.fbm5, erm=self.erm5)
            return e[0]
        self.backbone.blocks[3].register_forward_hook(fw_hook5)

        # PCoder number 6
        pmodule = Sequential(ConvTranspose2d(112, 80, kernel_size=3, stride=1, padding=1))
        self.pcoder6 = PCoder(pmodule, True, self.random_init)
        def fw_hook6(m, m_in, m_out):
            e = self.pcoder6(ff=m_out, fb=self.pcoder7.prd, target=self.pcoder5.rep, build_graph=self.build_graph, ffm=self.ffm6, fbm=self.fbm6, erm=self.erm6)
            return e[0]
        self.backbone.blocks[4].register_forward_hook(fw_hook6)

        # PCoder number 7
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear'),ConvTranspose2d(192, 112, kernel_size=3, stride=1, padding=1))
        self.pcoder7 = PCoder(pmodule, True, self.random_init)
        def fw_hook7(m, m_in, m_out):
            e = self.pcoder7(ff=m_out, fb=self.pcoder8.prd, target=self.pcoder6.rep, build_graph=self.build_graph, ffm=self.ffm7, fbm=self.fbm7, erm=self.erm7)
            return e[0]
        self.backbone.blocks[5].register_forward_hook(fw_hook7)

        # PCoder number 8
        pmodule = Sequential(ConvTranspose2d(320, 192, kernel_size=3, stride=1, padding=1))
        self.pcoder8 = PCoder(pmodule, False, self.random_init)
        def fw_hook8(m, m_in, m_out):
            e = self.pcoder8(ff=m_out, fb=None, target=self.pcoder7.rep, build_graph=self.build_graph, ffm=self.ffm8, fbm=self.fbm8, erm=self.erm8)
            return e[0]
        self.backbone.blocks[6].register_forward_hook(fw_hook8)


